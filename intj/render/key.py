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
class Operand:
    """A value a lowered `Step` reads or writes."""

    #: "temp" (a step's result), "int" (a literal), "int64_min", "arg" (a
    #: call argument), or "dep" or "comp" (a slot of that array)
    kind: str
    index: int = 0  # the temp's number, the literal, the argument or the slot


@dataclasses.dataclass(frozen=True)
class Step:
    """One statement of a lowered `grid_cpp` or heuristic.

    `heuristic.lower` and `grid.compile_grid` lower Python expressions to
    steps; the `lowered` macro in entry.c.jinja writes each as C.  `op` says
    what it does and which fields it reads:

    - "declare": `out` (`= a`), and `out2 = b` if set.
    - "assign": `out = a`, and `out2 = b` if set.
    - "not": `out = !a`.  "compare": `out = a <fn> b`, `fn` one of "eq",
      "ne", "lt", "le", "gt", "ge".  "min": the smaller of `a` and `b`.
    - "is_none": whether `a` (an "arg", else a 0/1 "int") is None; `negate`.
    - "pick": Python's `min` (`fn` "lt") or `max` ("gt") of values `a` and
      `c`, whose kinds are `b` and `d`, into `out` and `out2`.
    - "checked": `out = intj_grid_<fn>(a[, b])`, Python's integer arithmetic.
    - "heur_arg": the int or bool argument `a`, named `name`, into the value
      `out` and its kind `out2`.  "heur_tensor": tensor `a`'s method `fn`,
      with the literal index `b` if set.  "heur_dtype": tensor `a`'s dtype.
    - "grid_input": argument `a`, named `name`, as an `fn` ("int" or "bool").
      "grid_input_dep": the tuned value `a` of kind `b`, as an `fn`.
    - "grid_output": `a` as grid dimension `index`.
    - "if": `then` if `a` (not `a`, under `negate`), else `orelse` if set.
    """

    op: str
    out: Operand | None = None
    out2: Operand | None = None
    a: Operand | None = None
    b: Operand | None = None
    c: Operand | None = None
    d: Operand | None = None
    fn: str = ""
    name: str = ""
    index: int = 0
    negate: bool = False
    then: tuple[Step, ...] = ()
    orelse: tuple[Step, ...] | None = None


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
    #: each level's computed keys, lowered: its `intj_tuned_level_<n>`
    lowered: tuple[tuple[Step, ...], ...]


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
    kernel its launchers compile has to appear here: two launchers
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
    #: `render.module.intj_sources()`: intj's Python sources, by content.  The
    #: render is written in them, and the version is one of them.
    intj_sources: str
    #: `assume_constant_globals=True`: the globals the kernel reads, by value.
    #: triton's `cache_key` hashes only the `tl.constexpr` ones; a plain global
    #: (a tuple read by a `constexpr_function`) would otherwise let two
    #: different values share a module despite describing different globals.
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
    #: STATIC_COMPILE only: what the binary was compiled against -- the full
    #: `torch.__version__`, since a nightly or a patch release can move what the
    #: headers declare. The other modes discover everything at load, so their
    #: `.so` is torch-version-independent and these stay None -- which is what
    #: keeps them out of the digest.
    torch_version: str | None
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
    grid_arg_num: int | None = None
    grid_py_mode: bool = False
    grid_cpp: tuple[Step, ...] | None = None  # `grid_cpp`, lowered
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
