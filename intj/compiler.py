"""Compiles a launcher's kernel on cache misses and returns the record C launches.

`KernelCompiler` turns a miss into a record, for ordinary launchers and tuned
candidates alike. Triton reaches it through one port, `triton_load`: tests
replace it to count or fake compiles, and intj's own refusals still apply to
whatever it returns.
"""

from __future__ import annotations

import dataclasses
import threading
import weakref
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING, Any

from . import _imports
from . import backend as backends
from . import compile_options
from .errors import UnsupportedKernel
from .render.key import Param
from .specialization import CompilerInput, compiler_input

if TYPE_CHECKING:
    from .launcher import LaunchFunction
    from .plan import BuildPlan

# triton ships no type information
JitFunction = Any
CompiledKernel = Any

#: The port to Triton: compile `compiler_input` for `target` under the
#: canonical options and load the binary on the current device.
TritonLoad = Callable[[JitFunction, CompilerInput, Any, Any], CompiledKernel]


def triton_load(
    jit_func: JitFunction,
    source: CompilerInput,
    target: Any,
    canonical_options: Any,
) -> CompiledKernel:
    """Compiles with Triton and loads the binary on the current device."""
    _imports.ensure_triton_imported()
    # through the module, so a patched `triton.compiler.compile` is the one called
    kernel = _imports.triton.compiler.compile(
        source.ast_source(jit_func),
        target=target,
        options=canonical_options.__dict__,
    )
    kernel._init_handles()
    return kernel


def _refuse_unlaunchable(kernel: CompiledKernel) -> None:
    """Refuses a compiled kernel intj cannot launch."""
    md = kernel.metadata
    if md.num_ctas != 1:
        raise UnsupportedKernel("intj: num_ctas > 1 is not supported")
    if md.launch_cooperative_grid:
        raise UnsupportedKernel("intj: launch_cooperative_grid is not supported")
    if getattr(md, "launch_pdl", False):  # nvidia only
        raise UnsupportedKernel("intj: launch_pdl is not supported")
    if getattr(md, "global_scratch_size", 0) or md.profile_scratch_size:
        raise UnsupportedKernel(
            "intj: kernels requiring scratch memory are not supported"
        )


def _split_dynamic(
    names: Sequence[str], values: Sequence[object]
) -> tuple[dict[str, object], dict[str, object]]:
    """Splits one call's dynamic values into (compile options, knob paths)."""
    pairs = list(zip(names, values))
    return (
        {n: v for n, v in pairs if not n.startswith("knobs.")},
        {n: v for n, v in pairs if n.startswith("knobs.")},
    )


def _check_live_knobs(values: Mapping[str, object]) -> None:
    """Checks that each declared knob's passed value is its live one.

    A declared knob only keys: the miss compiles under the live knobs, and
    intj never sets one.

    Raises:
        ValueError: A passed value differs from the knob's current value.
    """
    _imports.ensure_triton_imported()
    for path, passed in values.items():
        _, group, name = path.split(".")
        live = getattr(getattr(_imports.knobs, group), name)
        if passed != live:
            raise ValueError(
                f"intj: dynamic knob {path!r} passed {passed!r} but the current "
                f"value is {live!r}"
            )


def _check_device(device: int) -> None:
    # _init_handles() loads the binary on the *current* device; launching that
    # function on another device's stream is a wrong-context launch
    current = backends.current_device()
    if device != current:
        raise UnsupportedKernel(
            f"intj: launching on device {device} while device {current} is current; "
            "make the target device current before the first launch"
        )


@dataclasses.dataclass(frozen=True)
class _Identity:
    """What one compile is cached and self-checked by.

    Knobs change the binary even where they feed no option.
    """

    compiler_input: CompilerInput
    options_hash: str
    knobs: tuple[tuple[str, object], ...]

    @classmethod
    def of(
        cls,
        compiler_input: CompilerInput,
        canonical: Any,
        knobs: Mapping[str, object],
    ) -> _Identity:
        return cls(compiler_input, canonical.hash(), tuple(sorted(knobs.items())))


def record(kernel: CompiledKernel, nparams: int) -> tuple[int, int, int, int]:
    """Returns `kernel`'s record for C: function, block_dim, shared, nparams.

    Raises:
        RuntimeError: Triton compiled another argument count than C packed.
    """
    md = kernel.metadata
    expected = sum(1 for ty in kernel.src.signature.values() if ty != "constexpr")
    if expected != nparams:
        raise RuntimeError(
            f"intj: packed {nparams} kernel arguments but triton compiled {expected}; "
            "this is an intj bug"
        )
    return (kernel.function, md.warp_size * md.num_warps, md.shared, nparams)


class KernelCompiler:
    """Compiles `jit_func` for cache misses, once per identity and device.

    An ordinary launcher stores one in `_compile_state` (`for_build`); a tuned
    launcher owns one for its candidates (`for_candidates`). It holds the
    compiled kernels, their lock, the identity each key compiled, and the
    launcher's compile override.
    """

    def __init__(
        self,
        jit_func: JitFunction,
        options: Mapping[str, Any] | None = None,
        knob_values: Mapping[str, object] | None = None,
        *,
        return_compiled: bool = False,
        params: Sequence[Param] = (),
        baked_values: Mapping[int, object] | None = None,
        dynamic: Sequence[str] = (),
        load: TritonLoad | None = None,
    ) -> None:
        self.jit_func = jit_func
        self.target = backends.current_target() if jit_func is not None else None
        self.backend = (
            _imports.make_backend(self.target)
            if self.target is not None
            else backends.HOST
        )
        self._options = dict(options or {})
        self._knob_values = dict(knob_values or {})
        # weak where the records own their CompiledKernel (return_compiled):
        # the cache must not keep it alive past them
        self._compiled: MutableMapping[tuple[object, ...], CompiledKernel] = (
            weakref.WeakValueDictionary() if return_compiled else {}
        )
        # Reentrant: a compile can launch this kernel again (a nested miss on
        # the same thread).
        self._lock = threading.RLock()
        self.seen: dict[bytes, object] = {}
        self.override: Callable[..., Any] | None = None
        self._params = params
        self._baked_values = baked_values
        self._return_compiled = return_compiled
        self._dynamic = dynamic
        # None: `triton_load`, looked up on each compile so a test can replace it
        self._load = load
        fed = (
            {
                p.name: p
                for p in params
                if p.name in compile_options.parameter_options(self.backend, params)
            }
            if self.target is not None
            else {}
        )
        self._fixed = {
            n: (baked_values or {})[p.index]
            for n, p in fed.items()
            if p.call_index is None
        }
        self._per_call = {
            n: p.call_index for n, p in fed.items() if p.call_index is not None
        }
        # Without per-call values every record has the same options.
        self._fixed_options = (
            self._canonical(self._fixed, {})
            if self.target is not None and not (dynamic or self._per_call)
            else None
        )

    @classmethod
    def host(cls) -> KernelCompiler:
        """Returns a compiler that compiles nothing: a host build's."""
        return cls(None)

    @classmethod
    def for_build(cls, b: BuildPlan) -> KernelCompiler:
        """Returns the compiler of an ordinary launcher's misses, for its checked build."""
        _, target = backends.select(b.option.no_gpu)
        if target is None:
            return cls.host()
        assert b.knob_values is not None, "GPU builds read knobs at the first call"
        return cls(
            b.jit_func,
            dict(b.compile_options),
            b.knob_values,
            return_compiled=b.option.return_compiled,
            params=b.layout.params,
            baked_values={
                p.index: p.baked for p in b.params if p.annotation.baked_value
            },
            dynamic=b.dynamic,
        )

    @classmethod
    def for_candidates(
        cls,
        jit_func: JitFunction,
        options: Mapping[str, Any],
        knob_values: Mapping[str, object],
        params: Sequence[Param],
        dynamic: Sequence[str],
    ) -> KernelCompiler:
        """Returns the compiler of a tuned launcher's candidates.

        Every parameter is public: the tuning chain sees the full Triton call.
        Its cache is strong: a `CompiledKernel` is loaded on one device, and
        records point into candidates no module sees.
        """
        return cls(jit_func, options, knob_values, params=params, dynamic=dynamic)

    def _canonical(
        self, options: Mapping[str, Any], knobs: Mapping[str, object]
    ) -> Any:
        """Returns the target's options: `options` over the base ones, then knobs."""
        return compile_options.resolve(
            self.target,
            self.jit_func,
            self._options,
            options,
            {**self._knob_values, **knobs},
        )

    def _compile(self, identity: _Identity, canonical: Any) -> CompiledKernel:
        """Returns the kernel for `identity` on the current device, compiled once."""
        key = (identity, backends.current_device())
        with self._lock:
            kernel = self._compiled.get(key)
            if kernel is None:
                load = self._load if self._load is not None else triton_load
                fresh = load(
                    self.jit_func, identity.compiler_input, self.target, canonical
                )
                _refuse_unlaunchable(fresh)
                # a nested miss may have stored one meanwhile; its record may
                # already be cached, so its kernel is the one that must stay
                kernel = self._compiled.setdefault(key, fresh)
        return kernel

    def split_call(
        self, device: int, args: tuple[Any, ...]
    ) -> tuple[dict[str, object], dict[str, object], tuple[Any, ...]]:
        """Splits a miss's arguments into its dynamic options, its knobs, and the rest.

        Raises:
            UnsupportedKernel: `device` is not the current device.
            ValueError: A declared knob was passed other than its live value.
        """
        _check_device(device)
        dynamic = self._dynamic
        options, knobs = _split_dynamic(dynamic, args[: len(dynamic)])
        _check_live_knobs(knobs)
        return options, knobs, args[len(dynamic) :]

    def miss(self, keyblob: bytes, nparams: int, device: int, *args: Any) -> Any:
        """Returns the record for one miss: the override's, else `compile_miss`'s."""
        if self.override is not None:
            return self.override(keyblob, nparams, device, *args)
        return self.compile_miss(keyblob, nparams, device, *args)

    def compile_miss(
        self, keyblob: bytes, nparams: int, device: int, *args: Any
    ) -> Any:
        """Compiles or reuses the kernel for one miss and returns its record."""
        if self.target is None:
            return 0, 1, 0, nparams
        dyn_options, dyn_knobs, args = self.split_call(device, args)
        canonical = (
            self._fixed_options
            if self._fixed_options is not None
            else self._canonical(
                {
                    **self._fixed,
                    **dyn_options,
                    **{n: args[i] for n, i in self._per_call.items()},
                },
                dyn_knobs,
            )
        )
        source = compiler_input(
            self.jit_func,
            self._params,
            args,
            self.backend,
            baked_values=self._baked_values,
        )
        identity = _Identity.of(source, canonical, dyn_knobs)
        # b"" is a keyless launcher's one key; seen is per launcher.
        if self.seen.setdefault(keyblob, identity) != identity:
            raise RuntimeError(
                "intj: one spec key maps to two annotated ASTSource inputs; this is an intj bug"
            )
        kernel = self._compile(identity, canonical)
        result = record(kernel, nparams)
        return (*result, kernel) if self._return_compiled else result

    def candidate(
        self,
        values: Mapping[str, Any],
        config_options: Mapping[str, Any],
        knobs: Mapping[str, object],
    ) -> CompiledKernel:
        """Returns the kernel for one tuned candidate, compiling it once.

        `values` holds every parameter's value by name.
        """
        canonical = self._canonical(
            {**config_options, **{n: values[n] for n in self._per_call}}, knobs
        )
        source = compiler_input(
            self.jit_func,
            self._params,
            [values[p.name] for p in self._params],
            self.backend,
        )
        return self._compile(_Identity.of(source, canonical, knobs), canonical)


def override_compile(
    launcher: LaunchFunction, fn: Callable[..., Any]
) -> Callable[[], None]:
    """Overrides one ordinary launcher's misses and returns a restore function.

    For tests and benchmarks. Builds `launcher` if needed. `fn` is called as
    `fn(keyblob, nparams, device, *args)` on subsequent cache misses. Sibling
    launchers are unaffected. The original compiler remains alive so its
    cached kernels' function handles stay valid.

    Returns:
        A function taking no arguments that restores the previous override.

    Raises:
        TypeError: The launcher owns a tuning chain.
    """
    holder = launcher.__self__
    holder.build()
    state = holder._compile_state  # pyright: ignore[reportPrivateUsage]  # launcher internals
    if not isinstance(state, KernelCompiler):
        raise TypeError("intj: compile overrides require an ordinary launcher")
    previous = state.override
    state.override = fn

    def restore() -> None:
        state.override = previous

    return restore
