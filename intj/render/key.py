"""`RenderKey`: everything a rendered module depends on, and the values it holds."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from ..annotation import CanonicalAnnotation, DeviceBinding


@dataclasses.dataclass(frozen=True)
class Param:
    """One declared kernel parameter, as the render needs it."""

    name: str
    index: int
    call_index: int | None
    annotation: CanonicalAnnotation
    memo: int | None = None  # this slot's intj_memo, if it takes objects


@dataclasses.dataclass(frozen=True)
class TuningRender:
    """What the render needs of an autotune/heuristics launcher."""

    levels: int  # lookups per launch
    computed: tuple[int, ...]  # computed keys keyed at each level
    #: level-0 computed keys: (value, kind) byte offsets in the level-0 key
    computed_fields: tuple[tuple[int, int], ...]
    computed_names: tuple[str, ...]  # level-major; the callback's `computed`
    deps: tuple[int, ...]  # dependent values C reads, stored per level
    dep_names: tuple[str, ...]  # level-major; the callback's `deps`
    meta_names: tuple[str, ...]  # grid_py: tuned parameters, declaration order
    source: str  # generated `intj_tuned_level_<n>` functions


@dataclasses.dataclass(frozen=True)
class DynamicSlot:
    """One `dynamic_options` entry, as the render needs it."""

    name: str  # compile option or `knobs.<group>.<name>`
    value_offset: int | None  # None only under an invariant test's mutation
    kind_offset: int | None
    memo: int  # its intj_memo in the launcher's trailing arrays


@dataclasses.dataclass(frozen=True)
class RenderKey:
    """Everything the rendered `.so` depends on, hashed into its digest.

    Anything that changes the generated code, the compiled binary, or which
    kernel the module's callback compiles has to appear here: two launchers
    whose keys match share one `.so`, so a missing field means a stale module.
    `RenderContext.from_key` derives what `entry.c.jinja` renders from it.
    """

    # What the render does not read: it is a pure function of the template,
    # the runtime headers and the fields after these.
    template: str  # entry.c.jinja bytes, hex
    runtime_header: str  # runtime/*.h bytes, hex
    cache_key: str  # jit_func.cache_key plus AST language: source and callees
    target: tuple[Any, ...]  # backend, arch, warp size
    options: str  # canonicalized compile options, hashed by triton
    triton: tuple[Any, ...]  # triton version and libtriton identity
    compiler: tuple[str, ...]  # the C compiler triton would invoke
    build_flags: tuple[
        str, ...
    ]  # changes to compiler flags invalidate cached .so files
    ext_suffix: str | None
    intj_version: tuple[int, int]
    #: `assume_constant_globals=True`: the globals the kernel reads, by value.
    #: triton's `cache_key` hashes only the `tl.constexpr` ones; a plain global
    #: (a tuple read by a `constexpr_function`) would otherwise let two
    #: different values share a module -- and its compile callback's kernel.
    global_values: tuple[tuple[str, str, tuple[object, ...]], ...]

    # What the render reads, through `RenderContext`.
    module_name: str
    kernel_repr: str
    params: tuple[Param, ...]
    nwords: int  # spec-key length, in uint64 words
    device_binding: DeviceBinding
    device_offset: int | None
    verify_annotation: bool
    max_slots: int  # kernel param slots, upper bound
    spec_pointer_range: int  # 1 if the backend specializes pointers on a 2 GiB range
    driver_path: str
    launch_symbol: str
    device_symbol: str
    error_symbol: str
    error_style: str  # "return" | "outparam"
    torch_access_mode: str  # resolved TorchAccessMode value
    kernel_cache: str  # KernelCache value: which hash map holds the kernels
    #: where that map comes from, when it is not intj's own. In the digest
    #: because the module is built against it.
    cache_include_dirs: tuple[str, ...]
    cache_archives: tuple[str, ...]
    #: STATIC_COMPILE only: what the binary was compiled against. The other modes
    #: discover everything at load, so their `.so` is torch-version-independent
    #: and these stay None -- which is what keeps them out of the digest.
    torch_version: tuple[int, int] | None
    cxx_abi: int | None
    #: The interpreter the module is built for and loaded into.  Every mode reads
    #: CPython internals (see intj/python_intf), so a module is never shared
    #: across versions, patch releases included; the render refuses to compile
    #: against any other `Python.h`.  ABI flags (`t`, `d`) are also in `ext_suffix`.
    python_version: tuple[int, int, int]
    free_threaded: bool
    cpython_static_compile_header: str  # intj/python_intf header for that version
    # Explicit pointer names -> live torch ScalarType codes. Their compact key
    # indices still come from the runtime ABI table, in every access mode.
    pointer_types: tuple[tuple[str, int], ...] = ()
    grid_arg: int | None = None
    grid_py_mode: bool = False
    grid_cpp_source: str = ""
    grid_extra: tuple[str, ...] = ()
    return_compiled: bool = False
    ndebug: bool = True  # NDEBUG for the whole translation unit
    tuning: TuningRender | None = None
    #: `dynamic_options`, in call order: they shape the call and the key
    dynamic: tuple[DynamicSlot, ...] = ()

    def digest(self) -> str:
        """Returns the SHA-256 hex digest of every field, independent of field order."""
        # sort_keys so the digest does not depend on field order
        blob = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()
