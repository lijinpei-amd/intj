"""Make the C launcher for a `@triton.jit` kernel; `intj.render` renders and builds it."""

from __future__ import annotations

import dataclasses
import functools
import threading
import types
from collections.abc import (
    Callable,
    Iterable,
    Mapping,
)
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, overload

from . import _imports, launcher_base
from . import backend as backends
from .annotation import (
    _canonical_value,  # pyright: ignore[reportPrivateUsage]  # tagged compiler constants
)
from .compile_options import knob_options, live_knobs
from .compiler import KernelCompiler
from .compiler import override_compile as override_compile
from .errors import UnsupportedKernel as UnsupportedKernel
from .options import OPTION_FIELDS, MakeLauncherKwargs, MakeLauncherOption
from .plan import BuildPlan, build_plan, is_chain
from .render import module as render_module
from .render.build import compiler_path

# triton and torch ship no type information, so everything reaching into them is
# typed `Any` on purpose.
JitFunction = Any

if TYPE_CHECKING:
    from typing_extensions import Unpack


def _delayed_keywords(delayed: Callable[[], Mapping[str, Any]]) -> dict[str, Any]:
    values = delayed()
    if not isinstance(values, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]  # user code
        raise TypeError(
            "intj: delayed must return a mapping of make_launcher keywords, "
            f"got {type(values).__name__}"
        )
    values = cast("Mapping[object, object]", values)
    unknown = sorted(
        repr(k) for k in values if type(k) is not str or k not in OPTION_FIELDS
    )
    if unknown:
        raise TypeError(
            "intj: delayed returned unknown make_launcher keyword(s) "
            f"[{', '.join(unknown)}]"
        )
    return cast("dict[str, Any]", dict(values))


def _resolved(option: MakeLauncherOption) -> MakeLauncherOption:
    """Returns `option` with `delayed` called and its keywords replacing the others."""
    if option.delayed is None:
        return option
    return dataclasses.replace(
        option, **_delayed_keywords(option.delayed), delayed=None
    )


def _chained(
    first: Callable[[], Mapping[str, Any]] | None,
    then: Callable[[], Mapping[str, Any]],
) -> Callable[[], Mapping[str, Any]]:
    if first is None:
        return then

    def chained() -> Mapping[str, Any]:
        return {**_delayed_keywords(first), **_delayed_keywords(then)}

    return chained


class Launcher:
    """A launcher's object: `k.__self__` for `k = make_launcher(...)`.

    `k` is a C function whose entry is a build shim until its first call
    succeeds, then the rendered launch.  This object holds that entry, the
    options and, once built, the module's per-launcher state.  The C base
    (`_intj_launcher_base.Launcher`) defines the `_C_*` attributes and methods,
    and runs `_build_body` on the first call; `check()` and the in-place edits
    run under its build lock through `_C_locked`. Every edit comes in two forms:
    `foo` returns a new, unbuilt launcher and `_foo` changes this one in place
    until it is built.
    """

    __slots__ = ()

    if TYPE_CHECKING:
        _compile_state: object

        @property
        def _C_options(self) -> MakeLauncherOption: ...
        @property
        def _C_module(self) -> types.ModuleType | None: ...
        @property
        def _C_built(self) -> bool: ...
        def _C_build(self) -> types.ModuleType: ...
        def _C_spec_key(self, *args: object) -> tuple[bytes, int]: ...
        def _C_key_chain(self, *args: object) -> tuple[tuple[bytes, ...], bool]: ...
        def _C_locked(self, fn: Callable[[], object], /) -> object: ...
        def _C_set_options(self, options: MakeLauncherOption, /) -> None: ...

    @property
    def options(self) -> MakeLauncherOption:
        """The launcher's options; once built, the resolved ones (`delayed` None)."""
        return self._C_options

    def check(self) -> None:
        """Runs every check the first call would, without building or storing anything.

        Calls `options.delayed` if set; the first call calls it again.  Does
        nothing once built.
        """
        if not self._C_built:
            self._C_locked(self._dry_run)

    def build(self) -> types.ModuleType:
        """Builds now if the first call has not, and returns the rendered module."""
        return self._C_build()

    def spec_key(self, *args: object) -> tuple[bytes, int]:
        """Returns `(key bytes, nparams)` for `(device, *dynamic, *public_args)`.

        Builds first.
        """
        return self._C_spec_key(*args)

    def key_chain(self, *args: object) -> tuple[tuple[bytes, ...], bool]:
        """Returns `((key bytes, ...), found)` for `(device, *dynamic, *public_args)`.

        One key per lookup level the call walks: an ordinary launcher's chain
        is its spec key alone.  Builds first.
        """
        return self._C_key_chain(*args)

    def _bind(self, /, **values: object) -> None:
        """Merges `values` into this launcher's `bind_values`; `bind` copies instead.

        Raises:
            RuntimeError: The launcher is built.
        """
        self._edit_options(
            lambda o: dataclasses.replace(
                o, bind_values={**(o.bind_values or {}), **values}
            )
        )

    def bind(self, /, **values: object) -> LaunchFunction:
        """Returns a new launcher with `values` merged into `bind_values`.

        Validates nothing: a missing, unknown or mistyped value fails in the
        new launcher's `check()`.  A bound tensor takes an exact
        `torch.Tensor`, `torch.nn.Parameter`, or None.
        """
        k = self._copy()
        k.__self__._bind(**values)
        return k

    def _bind_device(self, ordinal: int | None, /, **values: object) -> None:
        """Fixes this launcher to device `ordinal` in place; see `bind_device`.

        Raises:
            RuntimeError: The launcher is built.
        """
        self._edit_options(
            lambda o: dataclasses.replace(
                o, bind_device=ordinal, bind_values={**(o.bind_values or {}), **values}
            )
        )

    def bind_device(self, ordinal: int | None, /, **values: object) -> LaunchFunction:
        """Returns a new launcher fixed to device `ordinal`, `values` merged as for `bind`.

        It is called as `(stream, grid, ...)` and owns a separate kernel
        cache, or one kernel when no dynamic specialization key remains.
        None is `unbind_device()`.
        """
        k = self._copy()
        k.__self__._bind_device(ordinal, **values)
        return k

    def _unbind(self) -> None:
        """Drops this launcher's bound values in place; `unbind` copies instead."""
        self._edit_options(lambda o: dataclasses.replace(o, bind_values=None))

    def unbind(self) -> LaunchFunction:
        """Returns a new launcher with no bound values."""
        k = self._copy()
        k.__self__._unbind()
        return k

    def _unbind_device(self) -> None:
        """Makes this launcher take the device per call; `unbind_device` copies instead."""
        self._edit_options(lambda o: dataclasses.replace(o, bind_device=None))

    def unbind_device(self) -> LaunchFunction:
        """Returns a new launcher that takes the device per call."""
        k = self._copy()
        k.__self__._unbind_device()
        return k

    def _replace_delayed(self, fn: Callable[[], Mapping[str, Any]] | None, /) -> None:
        """Sets this launcher's `delayed` to `fn` (None drops it); `replace_delayed` copies instead.

        Raises:
            TypeError: `fn` is neither None nor callable.
            RuntimeError: The launcher is built.
        """
        if fn is not None and not callable(fn):
            raise TypeError(
                "intj: delayed must be None or a callable taking no arguments"
            )
        self._edit_options(lambda o: dataclasses.replace(o, delayed=fn))

    def replace_delayed(
        self, fn: Callable[[], Mapping[str, Any]] | None, /
    ) -> LaunchFunction:
        """Returns a new launcher whose `delayed` is `fn`; None drops it."""
        k = self._copy()
        k.__self__._replace_delayed(fn)
        return k

    def _chain_delayed(self, fn: Callable[[], Mapping[str, Any]], /) -> None:
        """Chains `fn` after this launcher's `delayed`; `chain_delayed` copies instead.

        Raises:
            TypeError: `fn` is not callable.
            RuntimeError: The launcher is built.
        """
        if not callable(fn):
            raise TypeError("intj: chain_delayed takes a callable taking no arguments")
        self._edit_options(
            lambda o: dataclasses.replace(o, delayed=_chained(o.delayed, fn))
        )

    def chain_delayed(self, fn: Callable[[], Mapping[str, Any]], /) -> LaunchFunction:
        """Returns a new launcher whose `delayed` also runs `fn`, whose keywords win."""
        k = self._copy()
        k.__self__._chain_delayed(fn)
        return k

    def _resolve_delayed(self) -> None:
        """Calls `delayed` now and applies its keywords in place; `resolve_delayed` copies instead.

        Does nothing when there is no `delayed`, built or not.
        """
        if self._C_options.delayed is None:
            return
        try:
            self._edit_options(_resolved)
        except RuntimeError:
            # another thread built it after the check above: a built launcher
            # holds resolved options, so there is nothing left to resolve
            if self._C_options.delayed is not None:
                raise

    def resolve_delayed(self) -> LaunchFunction:
        """Returns a new launcher with `delayed` called now and its keywords applied."""
        k = self._copy()
        k.__self__._resolve_delayed()
        return k

    def _copy(self) -> LaunchFunction:
        return _new_launcher(self._C_options)

    def _dry_run(self) -> None:
        # check()'s body, under the build lock: a first call may have built it since
        if not self._C_built:
            build_plan(_resolved(self._C_options))

    def _edit_options(
        self, edit: Callable[[MakeLauncherOption], MakeLauncherOption]
    ) -> None:
        # the `_foo` edits' body: options = edit(options) under the build lock
        def body() -> MakeLauncherOption:
            if self._C_built:
                raise RuntimeError(
                    "intj: a built launcher's options are fixed; the method "
                    "without the leading underscore returns an edited copy"
                )
            old = self._C_options
            self._C_set_options(edit(old))
            # dropped by _C_locked's caller, after the unlock: finalizers run Python
            return old

        self._C_locked(body)

    def _build_body(self) -> tuple[int, MakeLauncherOption, bytes]:
        """Builds this launcher and returns the entry, options and doc to publish.

        `_intj_launcher_base` calls it by name, under the build lock.

        Resolves `delayed`, checks, then queries the target, renders, compiles,
        loads, and fills this launcher's state block. Returns the rendered
        entry's address, the resolved options and the `__text_signature__`,
        for `_intj_launcher_base` to publish and swap in.

        `TRITON_INTERPRET=1` is refused here, on the first call: decoration
        never runs a kernel, so a module-level decorator does not make its
        module unimportable under the interpreter.
        Once this call succeeds the swapped-in entry never reads the knob
        again, so a launcher built with the interpreter off keeps launching if
        it is turned on later.
        """
        option = _resolved(self._C_options)
        b = build_plan(option)
        _imports.ensure_triton_imported()
        if _imports.knobs.runtime.interpret:
            raise UnsupportedKernel("intj: TRITON_INTERPRET=1 is not supported")
        self._check_target(b)
        module = render_module._build_module(b, _stub())  # pyright: ignore[reportPrivateUsage]  # renderer's build entry
        grid_py = (b.grid.fn, b.grid.hidden) if b.grid.mode == "py" else ()
        state = self._new_compile_state(b)
        address = module._init(self, _Interner(), *grid_py, *b.args)
        self._compile_state = state
        return address, option, b.doc

    def _check_target(self, b: BuildPlan) -> None:
        """Runs the build's checks that need the target, after every GPU-free one."""
        assert b.tuned is None, "make_launcher gives a chain a TunedLauncher"

    def _new_compile_state(self, b: BuildPlan) -> object:
        """Returns the Python compile state stored after initialization succeeds."""
        return KernelCompiler.for_build(b)

    def _compile_miss(
        self,
        keyblob: bytes,
        nparams: int,
        device: int,
        stream: int,
        controls: object,
        /,
        *args: Any,
    ) -> Any:
        """Returns the record for C to launch on an ordinary cache miss.

        `args` are the dynamic values, then the public kernel arguments.
        """
        del stream, controls
        state = cast(KernelCompiler, self._compile_state)
        return state.miss(keyblob, nparams, device, *args)


class LaunchFunction(Protocol):
    """What `make_launcher` returns: a C function whose `__self__` is its `Launcher`."""

    @property
    def __self__(self) -> Launcher: ...
    def __call__(self, *args: Any) -> Any: ...


def _new_launcher(options: MakeLauncherOption) -> LaunchFunction:
    stub, plain = _stub_and_type()
    # `delayed` cannot replace jit_func, so whether it is a chain is fixed now
    chain = is_chain(options.jit_func)
    return stub.new_launcher(_tuned_type() if chain else plain, options)


@overload
def make_launcher(
    jit_func: JitFunction,
    /,
    *,
    delayed: Callable[[], Mapping[str, Any]] | None = None,
    **kwargs: Unpack[MakeLauncherKwargs],
) -> LaunchFunction: ...


@overload
def make_launcher(
    *,
    delayed: Callable[[], Mapping[str, Any]] | None = None,
    **kwargs: Unpack[MakeLauncherKwargs],
) -> Callable[[JitFunction], LaunchFunction]: ...


#: make_launcher's "no kernel given" default: a positional None is an error, not
#: the decorator-factory form (an optional kernel whose import failed).
_NO_KERNEL: Any = object()


def make_launcher(
    jit_func: JitFunction = _NO_KERNEL,
    /,
    *,
    delayed: Callable[[], Mapping[str, Any]] | None = None,
    **kwargs: Unpack[MakeLauncherKwargs],
) -> Any:
    """Returns a fast launcher for `jit_func`, checked and built on first use.

    Called without `jit_func` (keyword arguments only), this is a decorator
    factory instead: `make_launcher(grid_cpp=grid)(kernel)` builds the same
    launcher `make_launcher(kernel, grid_cpp=grid)` would, so a kernel (plain
    or wrapped in `triton.autotune`/`triton.heuristics`) can be decorated
    directly:

        @intj.make_launcher(grid_cpp=grid)
        @triton.autotune(...)
        @triton.jit
        def k(...): ...

    Decoration only records a `MakeLauncherOption`: no kernel check, no GPU,
    no torch import.  Until one succeeds in this process, a call also imports
    triton and loads `_intj_launcher_base`, the C base every launcher uses,
    which can fail (see Raises); only the Triton refusal is remembered, and any
    other failure there is retried on the next call.  `k.__self__.check()`
    runs every other check now; otherwise the first call does, then builds the
    module (see docs/Usage.md, "Build on first call").

    The returned callable is a C function:

        launcher(device, stream, grid, arg0, arg1, ...)

    `device` is a device index in `[0, 256)` -- one byte of the spec key holds
    it -- and `stream` is a raw stream handle (both ints). The default `grid`
    is an int or a tuple/list of up to 3 ints. Declared `dynamic_options`
    values follow the grid controls. The remaining call arguments are the
    kernel's public parameters, positionally, in declaration order. Baked
    Argument and Constexpr values, and bound parameters, are omitted.

    Args:
        jit_func: A `@triton.jit` kernel, optionally wrapped in
            `triton.autotune`/`triton.heuristics`. Omit it for the decorator
            factory form.
        delayed: A function of no arguments returning more of these
            keywords, as a mapping; they replace the ones passed here. Each
            launcher calls it on its first call (and on every `check()`
            before that); its exception propagates and is retried on the
            next call.
        dynamic_options: Compile options and `knobs.<group>.<name>` paths
            passed per call, right after the grid controls, and keyed.
        extra_annotation: Parameter name -> annotation or shorthand: fixes
            types or specialization facts, bakes values, or marks values for
            binding.
        options: Triton compile options (`num_warps`, `num_stages`, ...)
            baked into every launch.
        torch_access_mode: How the module reads a tensor; None selects
            automatically. See `TorchAccessMode`.
        kernel_cache: The hash map behind the kernel cache; see `KernelCache`.
        no_gpu: Decode and cache on the host without compiling or launching a
            GPU kernel.
        verify_annotation: Check declared types, ranges and assumed facts;
            otherwise these are caller promises.
        bind_device: A non-negative int32 device ordinal: the launcher is
            called as `(stream, grid, ...)` and owns a separate kernel cache,
            or one kernel when no dynamic specialization key remains.
        bind_values: Bound parameter name -> value; see `Launcher.bind`.
        grid_arg: Take that many (1, 2 or 3) separate grid dimensions instead
            of `grid`.
        grid_cpp: An annotated def computing the grid in the extension from
            same-name JIT inputs and keyword-only extra inputs.
        grid_py: A callable given a fresh dict of JIT inputs on every launch,
            returning the grid. `grid_arg`, `grid_cpp` and `grid_py` are
            mutually exclusive.
        return_compiled: Return the cached Triton CompiledKernel after a
            successful launch, including on a zero-volume grid. Requires GPU
            mode.
        assume_constant_globals: Accept a kernel that reads global variables:
            their values when triton first hashed the kernel become part of
            the module identity, and are never checked again. Changing one
            afterwards is unsupported and not detected.
        ndebug: Compile the module with `NDEBUG` defined, so `assert()` in
            CPython's headers and intj's runtime is compiled out. False keeps
            them live, for debugging; it builds a separate module.

    Returns:
        The launcher, a C function whose `__self__` is its `Launcher`
        (`bind`, `bind_device`, `unbind`, `unbind_device`, `replace_delayed`,
        `chain_delayed`, `resolve_delayed`, each with an in-place `_` twin,
        and `check`). Without `jit_func`, a decorator that makes one.

    Raises:
        TypeError: A None kernel, an unknown keyword, or a `delayed` that is
            not callable.
        UnsupportedKernel: Triton is missing or older than 3.7, checked once
            per process: every later call raises the same refusal.  Or the
            host compiler failed to build `_intj_launcher_base`.
        ImportError: Importing an installed triton failed: a broken install,
            such as a missing dependency or extension module.
        RuntimeError: Triton finds no host C compiler.
        OSError: The C launcher base's build directory cannot be created.
    """
    unknown = kwargs.keys() - OPTION_FIELDS
    if unknown:
        raise TypeError(
            "intj: make_launcher() got unexpected keyword argument(s) "
            f"{sorted(unknown)}"
        )
    if delayed is not None and not callable(delayed):
        raise TypeError("intj: delayed must be None or a callable taking no arguments")
    if jit_func is _NO_KERNEL:
        return functools.partial(make_launcher, delayed=delayed, **kwargs)
    if jit_func is None:
        raise TypeError("intj: make_launcher got None instead of a kernel")
    option = MakeLauncherOption(jit_func, delayed=delayed, **kwargs)
    return _new_launcher(option)


_stub_loaded: tuple[types.ModuleType, type[Launcher]] | None = None
_tuned_loaded: type[Launcher] | None = None
_STUB_LOCK = threading.Lock()


def _over_stub(base: type[Launcher], stub: types.ModuleType) -> type[Launcher]:
    # Only the concrete class adds storage: both bases must remain layout-compatible.
    return cast(
        "type[Launcher]",
        type(
            base.__name__,
            (base, stub.Launcher),
            {"__slots__": ("_compile_state",), "__module__": base.__module__},
        ),
    )


def _stub_and_type() -> tuple[types.ModuleType, type[Launcher]]:
    """Returns the process's C launcher base and the `Launcher` type over it.

    Loaded beside the modules intj renders, once, under a lock: a second load
    has its own `_intj_launcher_base.Launcher`, and `new_launcher` refuses a
    type over another load's.
    """
    global _stub_loaded
    loaded = _stub_loaded
    if loaded is None:
        with _STUB_LOCK:
            loaded = _stub_loaded
            if loaded is None:
                _imports.ensure_triton_imported()
                stub = launcher_base.load_stub(
                    Path(_imports.knobs.cache.get_triton_dir("intj")) / "launcher_base",
                    compiler_path("c"),
                )
                loaded = _stub_loaded = (stub, _over_stub(Launcher, stub))
    return loaded


def _tuned_type() -> type[Launcher]:
    """Returns `TunedLauncher` over the process's C base, built on first use."""
    global _tuned_loaded
    tuned = _tuned_loaded
    if tuned is None:
        stub = _stub()
        with _STUB_LOCK:
            tuned = _tuned_loaded
            if tuned is None:
                from .tuning import TunedLauncher  # not at top: tuning imports this

                tuned = _tuned_loaded = _over_stub(TunedLauncher, stub)
    return tuned


def _stub() -> types.ModuleType:
    return _stub_and_type()[0]


def module_of(launcher: Callable[..., Any]) -> types.ModuleType:
    """Returns the rendered module behind `launcher`, building it if needed.

    Builds now if the launcher's first call has not.  For tests and debugging;
    `k.__self__.spec_key(...)` builds the same way.
    """
    return getattr(launcher, "__self__").build()


class _Interner:
    """One launcher's object values -> small ids, by canonical value.

    Per launcher, like its kernel cache: an id means one value within one
    launcher and nothing outside it.  Ids live for the process and never enter
    a `RenderKey`.  A JIT function's `cache_key` is read once, when its object
    is first seen, as triton does.  New ids are taken under this table's own
    lock rather than the launcher's C write lock: same effect, and the C side
    never holds a lock while calling Python.

    C asks this table only on a miss in its own object table, which keys
    what it has already answered: a str by content, an instance of
    `dtype_type` by its `name`, anything else by identity.
    """

    def __init__(self) -> None:
        _imports.ensure_triton_imported()
        self._ids: dict[tuple[object, ...], int] = {}
        self._lock = threading.Lock()
        self.dtype_type: type = _imports.tl.dtype  # read once by _init

    def __call__(self, value: object) -> int | None:
        try:
            canonical = _canonical_value(value)
        except ValueError:
            return None  # C raises the TypeError, naming the parameter
        found = self._ids.get(canonical)
        if found is None:
            with self._lock:  # two new values must not both take len(_ids)
                found = self._ids.setdefault(canonical, len(self._ids))
        return found


def triton_specialization(
    jit_func: JitFunction, args: Iterable[Any], options: Mapping[str, Any] | None = None
) -> list[tuple[str, Any]]:
    """Returns the `list[(type_str, key)]` triton would compute for these arguments."""
    binder = jit_func.device_caches[backends.current_device()][4]
    kwargs = knob_options(jit_func, options or {}, live_knobs())
    return binder(*args, **kwargs)[1]
