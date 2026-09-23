# Argument Annotations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add canonical argument annotations, compact typed specialization keys, baked and bound values, fixed-device launchers, optional low-overhead verification, and a GPU-free benchmark path to `intj.make_launcher()`.

**Architecture:** `intj/annotation.py` owns the public immutable values plus private merge and canonicalization rules. `launcher.py` turns one canonical parameter tuple into both the frozen `RenderContext` and the exact Triton `ASTSource`; the generated extension decodes only public call arguments, injects baked or bound values, and uses the module cache, a per-handle cache, or one nullable kernel pointer. Extend the existing runtime header and Jinja template instead of adding a second launcher stack.

**Tech Stack:** Python 3.12+, frozen dataclasses, Triton 3.8 `ASTSource`, Jinja2, CPython vectorcall C API, C11/C++20, pytest, pyright.

**Spec:** `docs/superpowers/specs/2026-09-23-argument-annotations-design.md`

## Global Constraints

- Preserve: **same intj cache key implies the same final annotated `ASTSource.signature`, `ASTSource.constants`, and `ASTSource.attrs`**.
- Do not mutate or clone the original Triton `JITFunction`.
- `RenderContext`, `ModuleKey`, `Param`, and nested fields stay frozen, hashable, and JSON-serializable. Use tuples, never lists.
- `RenderContext.params` contains only merged canonical annotations. Remove raw unmerged `ModuleKey.params`.
- Keep backend differences in `Backend` subclasses. Add device lookup as backend data; never branch on backend names in Python or Jinja.
- Keep generated modules out of `sys.modules` and keep multi-phase module initialization.
- Do not add nanobind or duplicate Python/C constants. Render compile-time values directly.
- Defaults remain `extra_annotation=None`, `bind_device=False`, `verify_annotation=False`, and `no_gpu=False`.
- `no_gpu=True` must not discover a GPU target, compile Triton GPU code, resolve a driver symbol, or launch a kernel.
- Preserve the error taxonomy: constructor kind errors are `TypeError`; malformed/conflicting/impossible annotations are `ValueError`; unsupported kernel shapes are `UnsupportedKernel`; checked call kind/range failures are `TypeError`, `ValueError`, or `OverflowError` as specified.
- CUDA is compile-checked only. Runtime tests use Triton 3.8.0 and torch 2.14+rocm7.2 on AMD.
- Run `PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests -q` and `pyright` before completion.
- Preserve the user's existing `.gitignore` modification. Stage only files named by each task.

## File Map

- Create `intj/_version.py`: sole package version source.
- Create `intj/annotation.py`: public values and private canonicalization.
- Create `tests/test_annotation.py`: GPU-independent constructor, merge, and layout tests.
- Modify `intj/__init__.py` and `pyproject.toml`: exports and dynamic versioning.
- Modify `intj/launcher.py`: annotation resolution, layout, factories, compiler inputs, and host mode.
- Modify `intj/runtime/intj_runtime.h`: decoded values, checks, packing, and owned tensor reads.
- Modify `intj/runtime/entry.c.jinja`: generated decode, binding, caches, device binding, and host mode.
- Modify `tests/test_launcher.py`: source, host-only, AMD runtime, invariant, binding, and CUDA compile tests.
- Modify `benchmarks/bench_launch.py` and `docs/Usage.md`: measured matrix and user documentation.

---

### Task 1: Use the package version as the cache schema

**Files:**
- Create: `intj/_version.py`
- Modify: `intj/__init__.py`
- Modify: `intj/launcher.py:40,96-120,201-216`
- Modify: `pyproject.toml:1-13`
- Test: `tests/test_launcher.py` beside the `ModuleKey` value tests

**Interfaces:**
- Produces: `intj._version.__version__: str`
- Produces: `intj.__version__: str`
- Produces: `ModuleKey.intj_version: tuple[int, int]`
- Removes: `SCHEMA_VERSION` and `ModuleKey.schema`

- [ ] **Step 1: Write the failing version tests**

Add:

```python
import tomllib

import intj


def test_package_version_has_one_source():
    data = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
    assert "version" not in data["project"]
    assert data["project"]["dynamic"] == ["version"]
    assert data["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "intj._version.__version__"
    }
    assert intj.__version__ == "0.1.0"


def test_module_key_uses_major_minor_version_only(monkeypatch):
    import intj.launcher as launcher

    monkeypatch.setattr(launcher, "__version__", "7.8.9")
    assert launcher._cache_version() == (7, 8)
    monkeypatch.setattr(launcher, "__version__", "7.8.10")
    assert launcher._cache_version() == (7, 8)
    monkeypatch.setattr(launcher, "__version__", "7.9.0")
    assert launcher._cache_version() == (7, 9)
```

Update `_module_key()` to include `intj_version=(0, 1)` and no `schema`.

- [ ] **Step 2: Confirm the tests fail**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_launcher.py -k 'package_version or major_minor' -q
```

Expected: FAIL because dynamic versioning and `_cache_version()` do not exist.

- [ ] **Step 3: Add the single version source**

Create `intj/_version.py`:

```python
"""The one source of intj's package version."""

__version__ = "0.1.0"
```

In `pyproject.toml`, replace the literal project version with:

```toml
[project]
name = "intj"
dynamic = ["version"]
description = "Incompatible Triton Jit: a low-overhead host launcher for triton kernels on AMD and NVIDIA GPUs"
requires-python = ">=3.12"
dependencies = ["jinja2", "triton>=3.8", "torch"]

[tool.setuptools.dynamic]
version = {attr = "intj._version.__version__"}
```

Keep all other existing sections unchanged.

- [ ] **Step 4: Replace `SCHEMA_VERSION`**

Import `__version__` in `launcher.py` and add:

```python
def _cache_version() -> tuple[int, int]:
    major, minor, *_ = __version__.split(".")
    return int(major), int(minor)
```

Replace `ModuleKey.schema` with `intj_version: tuple[int, int]` and pass `intj_version=_cache_version()` at its sole construction. Re-export `__version__` from `intj/__init__.py`.

- [ ] **Step 5: Verify and commit**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_launcher.py -k 'package_version or major_minor or module_key or value_types' -q
pyright
git add intj/_version.py intj/__init__.py intj/launcher.py pyproject.toml tests/test_launcher.py
git commit -m "Use package version in launcher cache keys"
```

Expected: tests PASS and pyright reports `0 errors`.

---

### Task 2: Add the immutable public annotation values

**Files:**
- Create: `intj/annotation.py`
- Create: `tests/test_annotation.py`
- Modify: `intj/__init__.py`

**Interfaces:**
- Produces: `Annotation`, `Argument`, `Constexpr`, `Specialization`, `AUTO`, `NEVER`, `Assume`
- Produces: `Fact`, `EqualTo`, `Aligned`, `PointerRange`, `BindValue`
- Produces: `INT_TYPES` and `FLOAT_TYPES`
- Keeps private: `_Unset`, `UNSET`, `_Auto`, and `_Never`

- [ ] **Step 1: Write constructor and export tests**

Create `tests/test_annotation.py`:

```python
# pyright: standard

import pytest
import triton.language as tl

import intj
from intj.annotation import UNSET


def test_public_annotation_exports_and_presets():
    assert intj.INT_TYPES == (tl.int32, tl.int64, tl.uint64)
    assert intj.FLOAT_TYPES == (tl.float32,)
    assert "UNSET" not in intj.__all__
    assert intj.Argument().type is UNSET
    assert intj.Argument(type=None).type is None


def test_annotation_sequences_are_immutable():
    arg = intj.Argument(type=[tl.int64, tl.int32, tl.int64])
    assert isinstance(arg.type, tuple)
    assert hash(arg) == hash(intj.Argument(type=(tl.int64, tl.int32, tl.int64)))


@pytest.mark.parametrize(
    "build,match",
    [
        (lambda: intj.Argument(type=[]), "type sequence must not be empty"),
        (lambda: intj.Argument(bind_value="tensor"), "bind_value"),
        (
            lambda: intj.Argument(value=1, bind_value=intj.BindValue.POINTER),
            "mutually exclusive",
        ),
        (lambda: intj.Assume(intj.EqualTo(2)), r"EqualTo\\(1\\)"),
        (lambda: intj.Assume(intj.Aligned(8)), r"Aligned\\(16\\)"),
        (lambda: intj.Assume(intj.PointerRange(64)), r"PointerRange\\(32\\)"),
    ],
)
def test_annotation_constructor_errors(build, match):
    with pytest.raises((TypeError, ValueError), match=match):
        build()
```

- [ ] **Step 2: Confirm import failure**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_annotation.py -q
```

Expected: collection FAIL because `intj.annotation` does not exist.

- [ ] **Step 3: Implement the public frozen values**

Create `intj/annotation.py` with these shapes:

```python
"""Argument annotations understood by intj launchers."""

import dataclasses
import enum
from collections.abc import Sequence
from typing import Any

import triton.language as tl


@dataclasses.dataclass(frozen=True)
class Annotation:
    pass


@dataclasses.dataclass(frozen=True)
class Specialization:
    pass


@dataclasses.dataclass(frozen=True)
class _Auto(Specialization):
    pass


@dataclasses.dataclass(frozen=True)
class _Never(Specialization):
    pass


AUTO = _Auto()
NEVER = _Never()


@dataclasses.dataclass(frozen=True)
class Fact:
    pass


@dataclasses.dataclass(frozen=True)
class EqualTo(Fact):
    value: int


@dataclasses.dataclass(frozen=True)
class Aligned(Fact):
    value: int


@dataclasses.dataclass(frozen=True)
class PointerRange(Fact):
    value: int


@dataclasses.dataclass(frozen=True, init=False)
class Assume(Specialization):
    facts: tuple[Fact, ...]

    def __init__(self, *facts: Fact) -> None:
        allowed = {EqualTo: 1, Aligned: 16, PointerRange: 32}
        for fact in facts:
            expected = allowed.get(type(fact))
            if expected is None:
                raise TypeError("intj: Assume accepts exact built-in Fact instances")
            if fact.value != expected:
                raise ValueError(
                    f"intj: only {type(fact).__name__}({expected}) is supported"
                )
        object.__setattr__(self, "facts", tuple(facts))


@dataclasses.dataclass(frozen=True)
class _Unset:
    pass


UNSET = _Unset()


class BindValue(enum.Enum):
    TENSOR = "tensor"
    POINTER = "pointer"


def _normalize_type_field(value: object) -> object:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = tuple(value)
        if not value:
            raise ValueError("intj: type sequence must not be empty")
    return value


@dataclasses.dataclass(frozen=True)
class Argument(Annotation):
    type: Any = UNSET
    specialize: Specialization | _Unset = UNSET
    value: object = UNSET
    bind_value: BindValue | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _normalize_type_field(self.type))
        if not isinstance(self.specialize, (Specialization, _Unset)):
            raise TypeError("intj: specialize must be AUTO, NEVER, or Assume(...)")
        if self.bind_value is not None and not isinstance(self.bind_value, BindValue):
            raise TypeError("intj: bind_value must be a BindValue or None")
        if self.value is not UNSET and self.bind_value is not None:
            raise ValueError("intj: value and bind_value are mutually exclusive")


@dataclasses.dataclass(frozen=True)
class Constexpr(Annotation):
    type: Any = UNSET
    power_of_two_or_zero: bool = False
    value: object = UNSET

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _normalize_type_field(self.type))
        if not isinstance(self.power_of_two_or_zero, bool):
            raise TypeError("intj: power_of_two_or_zero must be bool")


INT_TYPES = (tl.int32, tl.int64, tl.uint64)
FLOAT_TYPES = (tl.float32,)
```

- [ ] **Step 4: Re-export exactly the approved public surface**

Add these names to `intj.__all__`:

```python
"Annotation", "Argument", "Constexpr", "Specialization", "AUTO", "NEVER",
"Assume", "Fact", "EqualTo", "Aligned", "PointerRange", "BindValue",
"INT_TYPES", "FLOAT_TYPES", "__version__"
```

Do not export `UNSET`, `_Unset`, `_Auto`, or `_Never`.

- [ ] **Step 5: Verify and commit**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_annotation.py -q
pyright
git add intj/annotation.py intj/__init__.py tests/test_annotation.py
git commit -m "Add argument annotation values"
```

Expected: tests PASS and pyright reports `0 errors`.

---

### Task 3: Merge sources into one canonical annotation

**Files:**
- Modify: `intj/annotation.py`
- Modify: `intj/launcher.py:123-220,585-649`
- Test: `tests/test_annotation.py`

**Interfaces:**
- Produces: private frozen `CanonicalAnnotation` and `ResolvedParam`
- Produces: `_resolve_annotations(jit_func, extra_annotation) -> tuple[ResolvedParam, ...]`
- Produces: `_canonical_value(value) -> tuple[object, ...]`

- [ ] **Step 1: Add a file-backed JIT helper and merge tests**

Add:

```python
import dataclasses
import importlib.util
import json


def kernel_from_source(tmp_path, source):
    path = tmp_path / "annotated_kernel.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel


def test_annotation_sources_merge_field_by_field(tmp_path):
    from intj.annotation import _resolve_annotations

    kernel = kernel_from_source(
        tmp_path,
        "from intj import Argument\\n"
        "import triton\\n"
        "import triton.language as tl\\n\\n"
        "@triton.jit(do_not_specialize_on_alignment=['x'])\\n"
        "def kernel(x: Argument(type=tl.int32), y):\\n"
        "    pass\\n",
    )
    params = _resolve_annotations(
        kernel, {"x": intj.Argument(specialize=intj.NEVER)}
    )
    x = params[0].annotation
    assert x.types == ("i32",)
    assert (x.equal_to_one, x.aligned_16, x.pointer_range_32) == (
        "never", "never", "never"
    )


def test_none_and_unset_remain_distinct_after_merge(tmp_path):
    from intj.annotation import _resolve_annotations

    kernel = kernel_from_source(
        tmp_path,
        "import triton\\n\\n@triton.jit\\ndef kernel(x, y):\\n    pass\\n",
    )
    params = _resolve_annotations(kernel, {"x": None, "y": intj.Argument()})
    assert params[0].annotation.types == (None,)
    assert params[1].annotation.types is None


def test_equivalent_inputs_have_one_canonical_form(tmp_path):
    from intj.annotation import _resolve_annotations

    kernel = kernel_from_source(
        tmp_path,
        "import triton\\n\\n@triton.jit\\ndef kernel(x):\\n    pass\\n",
    )
    one = _resolve_annotations(
        kernel,
        {"x": intj.Argument(
            type=[tl.int64, tl.int32, tl.int64],
            specialize=intj.Assume(intj.Aligned(16), intj.EqualTo(1)),
        )},
    )
    two = _resolve_annotations(
        kernel,
        {"x": intj.Argument(
            type=[tl.int32, tl.int64],
            specialize=intj.Assume(intj.EqualTo(1), intj.Aligned(16)),
        )},
    )
    assert one == two
    assert hash(one) == hash(two)
    json.dumps(dataclasses.asdict(one[0].annotation), sort_keys=True)
```

Also add parameterized rejection tests for:

- unknown `extra_annotation` name;
- explicit `AUTO` versus a fixed type;
- `AUTO` versus decorator `NEVER`;
- `Argument` versus `Constexpr`;
- an external `Fact` subclass;
- a dead fact on every possible type;
- `EqualTo(1)` plus `Aligned(16)` on one integer branch;
- unsupported scalar/constexpr dtypes;
- tensor or pointer values passed through `Argument(value=...)`;
- `BindValue.POINTER` without exactly one explicit pointer type.

- [ ] **Step 2: Confirm canonicalization tests fail**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_annotation.py -k 'merge or canonical or unset or reject' -q
```

Expected: FAIL because canonicalization does not exist.

- [ ] **Step 3: Add the private canonical value types**

Add:

```python
@dataclasses.dataclass(frozen=True)
class KeyField:
    kind: str
    width: int
    offset: int = -1


@dataclasses.dataclass(frozen=True)
class CanonicalAnnotation:
    kind: str
    types: tuple[str | None, ...] | None
    equal_to_one: str
    aligned_16: str
    pointer_range_32: str
    facts: tuple[tuple[str, int], ...]
    power_of_two_or_zero: bool
    bind_value: str | None
    baked_value: tuple[object, ...]
    key_fields: tuple[KeyField, ...] = ()


@dataclasses.dataclass(frozen=True)
class ResolvedParam:
    name: str
    index: int
    annotation: CanonicalAnnotation
    baked: object = UNSET
```

`CanonicalAnnotation.types is None` means AUTO; `types == (None,)` means exact Python `None`. `baked_value == ()` means no baked value; `(\"none\",)` means baked `None`.

- [ ] **Step 4: Implement stable type and value canonicalization**

Use:

```python
def _canonical_value(value: object) -> tuple[object, ...]:
    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", str(value))
    if type(value) is float:
        return ("float64", struct.pack(">d", value).hex())
    raise ValueError("intj: baked values must be scalar int, float, bool, or None")
```

Canonicalize Triton dtypes through their compiler spellings (`i32`, `u64`, `fp32`, `*fp32`). Deduplicate and sort them with `None` before strings. Reject `AUTO` inside a sequence. Keep the original baked scalar only in `ResolvedParam.baked`; only its tagged tuple enters `RenderContext`.

- [ ] **Step 5: Implement field-wise merging**

Implement `_resolve_annotations()` in this order:

1. Read the raw `inspect.Parameter.annotation`, not Triton's normalized string. Accept exact intj annotations, Triton dtypes, bare `tl.constexpr`, bare `None`, and omission.
2. Merge `extra_annotation[name]` per field. Equal duplicates pass; unequal specified values raise `ValueError` naming the parameter and field.
3. Fold `do_not_specialize` into equality NEVER plus alignment NEVER. Fold `do_not_specialize_on_alignment` into alignment NEVER.
4. Expand specialization `AUTO` to three explicit AUTO modes, `NEVER` to three NEVER modes, and `Assume` only to its named ASSUME modes.
5. Canonicalize remaining omitted type/fact modes to AUTO.
6. Reject dead facts and every non-`None` type branch with no satisfying value. Do not delete an impossible branch.
7. Allow ordinary integer dtypes, `tl.uint1`, `tl.float32`, `tl.float64`, pointers, and exact `None`. Allow constexpr integers, `tl.uint1`, `tl.float64`, and exact `None`; reject fp8/fp16/bf16/fp32.
8. Validate baked values even when verification is disabled.
9. Require `Argument(value=...)` and both binding modes to resolve one effective type with no applicable AUTO fact.
10. Require `BindValue.POINTER` to have exactly one explicit pointer type.

Change `make_launcher()` to call this before target discovery and dependency provisioning. Remove the old blanket rejection of `extra_annotation` and non-constexpr annotations.

- [ ] **Step 6: Verify and commit**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_annotation.py -q
pyright
git add intj/annotation.py intj/launcher.py tests/test_annotation.py
git commit -m "Canonicalize merged argument annotations"
```

Expected: tests PASS and pyright reports `0 errors`.

---

### Task 4: Compute key fields from canonical parameters

**Files:**
- Modify: `intj/annotation.py`
- Modify: `intj/launcher.py:51-115,158-216,617-649`
- Modify: `tests/test_annotation.py`
- Modify: `tests/test_launcher.py:396-496,789-839`

**Interfaces:**
- Produces: private `DeviceBinding` with `FIXED="fixed"` and `NOT_FIXED="not_fixed"`
- Produces: `_layout_fields(fields, device_binding) -> KeyLayout`
- Produces: `_render_params(resolved, device_binding) -> tuple[tuple[Param, ...], int | None, int]`
- Changes: `Param` stores `name`, original `index`, public `call_index`, and `CanonicalAnnotation`
- Changes: `RenderContext` stores `device_binding`, `device_offset`, `verify_annotation`, and `no_gpu`

- [ ] **Step 1: Replace declaration-position layout tests**

Add:

```python
@pytest.mark.parametrize(
    "widths,fixed_device,expected_offsets,device_offset,nwords",
    [
        ((8, 4, 2, 1), False, (0, 8, 12, 14), 15, 2),
        ((1, 8, 1, 4, 2), False, (14, 0, 15, 8, 12), 16, 3),
        ((2, 2, 1), True, (0, 2, 4), None, 1),
        ((), True, (), None, 0),
    ],
)
def test_key_fields_are_grouped_by_descending_alignment(
    widths, fixed_device, expected_offsets, device_offset, nwords
):
    from intj.annotation import DeviceBinding, KeyField, _layout_fields

    layout = _layout_fields(
        tuple((index, KeyField("payload", width)) for index, width in enumerate(widths)),
        DeviceBinding.FIXED if fixed_device else DeviceBinding.NOT_FIXED,
    )
    assert tuple(field.offset for _, field in layout.fields) == expected_offsets
    assert layout.device_offset == device_offset
    assert layout.nwords == nwords
```

Add focused cases proving:

- untyped dynamic constexpr gets a descriptor byte and 8-byte payload;
- a typed constexpr allowlist gets a descriptor and a payload at its largest width;
- `power_of_two_or_zero=True` gets a 1-byte payload and no wide payload;
- exact dynamic `None` constexpr gets no field;
- an exact ordinary type with no AUTO fact gets no field;
- the dynamic device byte follows all parameter-derived 8-bit fields;
- declaration order is stable within each width group.

- [ ] **Step 2: Confirm layout tests fail**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_annotation.py -k layout -q
```

Expected: FAIL because `DeviceBinding` and `_layout_fields()` do not exist.

- [ ] **Step 3: Implement the stable descending-width layout**

Add:

```python
class DeviceBinding(enum.StrEnum):
    FIXED = "fixed"
    NOT_FIXED = "not_fixed"


@dataclasses.dataclass(frozen=True)
class KeyLayout:
    fields: tuple[tuple[int, KeyField], ...]
    device_offset: int | None
    nwords: int


def _layout_fields(
    fields: tuple[tuple[int, KeyField], ...],
    device_binding: DeviceBinding,
) -> KeyLayout:
    ordered = sorted(fields, key=lambda item: -item[1].width)
    placed: list[tuple[int, KeyField]] = []
    offset = 0
    for param_index, field in ordered:
        placed.append((param_index, dataclasses.replace(field, offset=offset)))
        offset += field.width
    device_offset = None
    if device_binding is DeviceBinding.NOT_FIXED:
        device_offset = offset
        offset += 1
    return KeyLayout(tuple(placed), device_offset, (offset + 7) // 8)
```

Derive unplaced fields with these exact rules:

- descriptor byte when runtime type or any AUTO fact can change compiler input;
- untyped dynamic constexpr: descriptor plus 8-byte payload;
- typed dynamic constexpr: payload at the largest allowed width, plus a descriptor for a multi-type allowlist;
- power-of-two-or-zero constexpr: exactly one 1-byte payload; the encoding uniquely determines the value and therefore the deterministic selected integer type;
- baked, bound, exact-`None` constexpr, and fully fixed ordinary parameter: no key field.

After global placement, replace each frozen annotation with its own placed `key_fields`.
`_render_params()` assigns `call_index` only to parameters that remain in the
public callable and returns `(params, device_offset, nwords)`.

- [ ] **Step 4: Replace render and module-key shapes**

Replace `Param` with:

```python
@dataclasses.dataclass(frozen=True)
class Param:
    name: str
    index: int
    call_index: int | None
    annotation: CanonicalAnnotation
```

Add to `RenderContext`:

```python
    device_binding: str
    device_offset: int | None
    verify_annotation: bool
    no_gpu: bool
```

Remove `header_words`. Keep `nwords=0` as a real logical no-key state. Remove `ModuleKey.params` because merged canonical params are already in `context.params`. Update `_render_context()` and `_module_key()` test fixtures accordingly.

- [ ] **Step 5: Rewrite the rendered-byte differential test**

Make `test_rendered_key_puts_every_byte_where_python_says` read expected offsets from `param.annotation.key_fields` and `context.device_offset`. Include 64-, 32-, 16-, and 8-bit payloads, and assert every final padding byte is zero.

- [ ] **Step 6: Verify and commit**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_annotation.py tests/test_launcher.py -k 'layout or value_types or digest' -q
pyright
git add intj/annotation.py intj/launcher.py tests/test_annotation.py tests/test_launcher.py
git commit -m "Lay out canonical specialization keys"
```

Expected: tests PASS and pyright reports `0 errors`.

---

### Task 5: Add the GPU-free host path

**Files:**
- Modify: `intj/launcher.py:123-220,484-614,674-719`
- Modify: `intj/runtime/entry.c.jinja`
- Test: `tests/test_launcher.py`

**Interfaces:**
- Adds: `make_launcher(..., no_gpu: bool = False)`
- Produces: `_make_host_compile_callback()` returning dummy kernel metadata
- Keeps: `entry(device, stream, grid, ...)` while the device remains dynamic

- [ ] **Step 1: Write host-only smoke and refusal tests**

Add:

```python
def test_no_gpu_skips_target_driver_compile_and_launch(monkeypatch):
    import intj.launcher as launcher

    def forbidden(*args, **kwargs):
        raise AssertionError("GPU path was reached")

    monkeypatch.setattr(launcher, "_current_target", forbidden)
    monkeypatch.setattr(launcher, "_current_device", forbidden)
    host = make_launcher(scale, no_gpu=True, torch_access=TorchAccess.CPYTHON)
    x = torch.ones(8)
    out = torch.empty(8)
    assert host(0, 0, (1,), x, out, 8, 2.0, 8) is None
    assert host(0, 0, (1,), x, out, 8, 2.0, 8) is None


def test_no_gpu_rejects_real_backend_options():
    with pytest.raises(UnsupportedKernel, match="no_gpu.*options"):
        make_launcher(scale, no_gpu=True, options={"num_warps": 8})
```

- [ ] **Step 2: Confirm host-only tests fail**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_launcher.py -k no_gpu -q
```

Expected: FAIL because `make_launcher()` has no `no_gpu` argument.

- [ ] **Step 3: Separate kernel validation from GPU discovery**

Make `_check_kernel()` validate only the JIT function, hooks, globals, parameter shapes, and forbidden option names. In `make_launcher()` use:

```python
if no_gpu and options:
    raise UnsupportedKernel("intj: no_gpu=True supports only default compile options")

if no_gpu:
    target = None
    backend = None
    canonical_options = None
else:
    target = _current_target()
    backend = BACKENDS.get(target.backend)
    if backend is None:
        raise UnsupportedKernel(
            f"intj: backend {target.backend!r} is unknown; "
            f"have: {', '.join(sorted(BACKENDS))}"
        )
    canonical_options = _canonical_options(target, options)
```

For host mode, put `target=("host-only",)` and `options="host-defaults"` in `ModuleKey`. Use the normal C compiler identity and normal cache backend provisioning.
Render empty driver strings and baseline AUTO pointer-range support as false:
without target discovery there is no backend whose optional range specialization
can be inferred. Explicit `Assume(PointerRange(32))` remains structural and can
still be exercised by host-only verification.

- [ ] **Step 4: Render a host-only cold path**

Add:

```python
def _make_host_compile_callback() -> Callable[..., tuple[int, int, int, int]]:
    def compile_callback(
        keyblob: bytes, nparams: int, device: int, *args: Any
    ) -> tuple[int, int, int, int]:
        del keyblob, device, args
        return 0, 1, 0, nparams

    return compile_callback
```

In `entry.c.jinja`, guard driver loading, symbol resolution, context checks, and the final driver call with `{% if not no_gpu %}`. Host mode still decodes, builds the key, performs cache lookup, invokes the dummy callback on a miss, installs the dummy record, and validates `nparams`. Return `None` instead of launching.

Define the physical C key array with at least one word:

```c
#define INTJ_NWORDS {{ [nwords, 1] | max }}
#define INTJ_HAS_KEY {{ 1 if nwords else 0 }}
```

The logical `RenderContext.nwords` remains zero for the later no-map path.

- [ ] **Step 5: Verify host and GPU defaults**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_launcher.py -k 'no_gpu or matches_triton' -q
pyright
```

Expected: host-only tests PASS; existing AMD launch tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add intj/launcher.py intj/runtime/entry.c.jinja tests/test_launcher.py
git commit -m "Add GPU-free launcher execution"
```

---

### Task 6: Decode ordinary annotated arguments and baked values

**Files:**
- Modify: `intj/runtime/intj_runtime.h:18-100,390-648,734-827`
- Modify: `intj/runtime/entry.c.jinja:1-250`
- Modify: `intj/launcher.py`
- Test: `tests/test_launcher.py`

**Interfaces:**
- Produces: `intj_decoded` and `intj_decode_argument(...)`
- Adds: `make_launcher(..., verify_annotation: bool = False)`
- Emits: one combined `INTJ_UNLIKELY` check and one following `INTJ_ASSUME` per checked constrained argument

- [ ] **Step 1: Write host-only ordinary-argument tests**

Add one local helper and the small kernels reused by Tasks 6-10:

```python
def _kernel_from_source(tmp_path, name, signature):
    import importlib.util

    path = tmp_path / f"{name}.py"
    path.write_text(
        "import triton\\n\\n"
        f"@triton.jit\\ndef {name}({signature}):\\n"
        "    pass\\n"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


@pytest.fixture
def scalar_kernel(tmp_path):
    return _kernel_from_source(tmp_path, "scalar_kernel", "x")


@pytest.fixture
def power_kernel(tmp_path):
    return _kernel_from_source(tmp_path, "power_kernel", "N")


@pytest.fixture
def bound_kernel(tmp_path):
    return _kernel_from_source(tmp_path, "bound_kernel", "x, p, n")


@pytest.fixture
def device_kernel(tmp_path):
    return _kernel_from_source(tmp_path, "device_kernel", "x")


@pytest.fixture
def pointer_kernel(tmp_path):
    return _kernel_from_source(tmp_path, "pointer_kernel", "x")
```

Then add ordinary tests through `extra_annotation`:

```python
@pytest.mark.parametrize(
    "annotation,good,bad",
    [
        (Argument(type=tl.int32, specialize=NEVER), 7, 2**31),
        (Argument(type=tl.uint1, specialize=NEVER), True, 1),
        (Argument(type=tl.float64, specialize=NEVER), 1.25, 1),
        (Argument(type=None, specialize=NEVER), None, 0),
        (
            Argument(type=(tl.int32, tl.float32), specialize=NEVER),
            7,
            None,
        ),
    ],
)
def test_checked_ordinary_types(annotation, good, bad, scalar_kernel):
    launch = make_launcher(
        scalar_kernel,
        extra_annotation={"x": annotation},
        verify_annotation=True,
        no_gpu=True,
    )
    assert launch(0, 0, 1, good) is None
    with pytest.raises((TypeError, ValueError, OverflowError), match="x"):
        launch(0, 0, 1, bad)
```

Add tests for:

- exact pointer annotation treating `None` and integer `0` as null pointers;
- bare unannotated `None` retaining constexpr-None behavior;
- `NEVER` removing equality, alignment, and pointer-range key variation;
- `Assume(EqualTo(1), Aligned(16))` rejecting the impossible branch at creation;
- `Assume(Aligned(16), PointerRange(32))` omitting those facts from the key;
- `Argument(value=...)` removing a scalar/`None` parameter from the public call and key;
- reordered equivalent annotations producing the same module path;
- structurally different annotations and `verify_annotation` values producing different module paths.

- [ ] **Step 2: Confirm the runtime tests fail**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_launcher.py -k 'checked_ordinary or baked_argument or annotation_module' -q
```

Expected: FAIL because the template still uses the old unannotated decoder.

- [ ] **Step 3: Refactor the existing decoder once**

Replace `INTJ_DECODE` with an always-inlined structural decoder that is shared by normal entry and bound handles:

```c
typedef enum {
  INTJ_VALUE_TENSOR,
  INTJ_VALUE_BOOL,
  INTJ_VALUE_I64,
  INTJ_VALUE_U64,
  INTJ_VALUE_FP64,
  INTJ_VALUE_NONE
} intj_value_kind;

typedef struct {
  intj_value_kind kind;
  uint64_t bits;
  void *pointer;
  int64_t storage_nbytes;
  uint8_t dtype_index;
} intj_decoded;

static INTJ_ALWAYS_INLINE int
intj_decode_argument(intj_state *st, PyObject *o, int want_size,
                     const char *pname, intj_decoded *out);
```

Reuse the current exact type gates, `intj_as_int`, three tensor readers, dtype-index validation, and parameter error chaining. Decode Python floats once as binary64; generated packing chooses fp32 or fp64 without rereading the object.

Add:

```c
#if defined(__clang__)
#define INTJ_ASSUME(x) __builtin_assume(x)
#elif defined(__GNUC__)
#define INTJ_ASSUME(x) do { if (!(x)) __builtin_unreachable(); } while (0)
#else
#define INTJ_ASSUME(x) ((void)0)
#endif
```

- [ ] **Step 4: Generate type selection, checking, key bytes, and slots**

For each public ordinary argument, Jinja must:

1. call `intj_decode_argument` once;
2. select AUTO type or the deterministic allowed type;
3. compute applicable AUTO/NEVER/ASSUME facts;
4. when checked, compute one `valid_<index>` boolean, branch once with `INTJ_UNLIKELY(!valid_<index>)`, report the exact failing constraint in that cold block, then call `INTJ_ASSUME(valid_<index>)`;
5. write descriptor bytes only at the offsets in `annotation.key_fields`;
6. append a GPU value slot unless the final signature is constexpr.

Use `memcpy` helpers for 16/32/64-bit little-endian key stores. Zero the full key first so final padding is deterministic. Do not emit semantic predicates at all when `verify_annotation=False`; retain structural checks required to read CPython objects safely.

Embed `Argument(value=...)` from its tagged canonical bits. It has no public argument and no key field, but append its fixed GPU slot unless its final specialization is constexpr.

- [ ] **Step 5: Assert generated check shape**

Add a source test that opens the generated `.c` or `.cpp` next to the extension:

```python
def test_checked_source_has_one_predicted_branch_per_argument(tmp_path):
    checked = make_launcher(
        scalar_kernel,
        extra_annotation={
            "x": Argument(type=tl.int32, specialize=Assume(Aligned(16)))
        },
        verify_annotation=True,
        no_gpu=True,
    )
    unchecked = make_launcher(
        scalar_kernel,
        extra_annotation={
            "x": Argument(type=tl.int32, specialize=Assume(Aligned(16)))
        },
        verify_annotation=False,
        no_gpu=True,
    )
    checked_src = pathlib.Path(checked.__self__.__file__).with_suffix(".c").read_text()
    unchecked_src = pathlib.Path(unchecked.__self__.__file__).with_suffix(".c").read_text()
    assert checked_src.count("INTJ_UNLIKELY(!valid_0)") == 1
    assert "INTJ_ASSUME(valid_0)" in checked_src
    assert "valid_0" not in unchecked_src
```

Choose `.cpp` when `TorchAccess.CXX` or a C++ cache backend is selected.

- [ ] **Step 6: Verify and commit**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_launcher.py -k 'ordinary or annotation or rendered_key or check_shape' -q
pyright
git add intj/launcher.py intj/runtime/intj_runtime.h intj/runtime/entry.c.jinja tests/test_launcher.py
git commit -m "Render ordinary argument annotations"
```

Expected: focused tests PASS and pyright reports `0 errors`.

---

### Task 7: Decode typed and compact constexpr values

**Files:**
- Modify: `intj/runtime/intj_runtime.h:828-860`
- Modify: `intj/runtime/entry.c.jinja`
- Modify: `intj/launcher.py`
- Test: `tests/test_launcher.py`

**Interfaces:**
- Produces: `intj_decode_constexpr(...)` using the same `intj_decoded` structure
- Produces: signed power-of-two-or-zero byte encoding
- Preserves: bare `tl.constexpr` descriptor plus 64-bit payload behavior

- [ ] **Step 1: Write constexpr key and validation tests**

Add:

```python
@pytest.mark.parametrize(
    "value,encoded",
    [
        (0, 0x00),
        (1, 0x01),
        (2, 0x02),
        (2**63, 0x40),
        (-1, 0x81),
        (-2, 0x82),
        (-2**63, 0xC0),
    ],
)
def test_power_of_two_or_zero_encoding(power_kernel, value, encoded):
    module = getattr(
        make_launcher(
            power_kernel,
            extra_annotation={
                "N": Constexpr(
                    type=(tl.int64, tl.uint64),
                    power_of_two_or_zero=True,
                )
            },
            verify_annotation=True,
            no_gpu=True,
        ),
        "__self__",
    )
    key, _ = module.spec_key(value)
    # One parameter: compact value byte, then the dynamic device byte.
    assert key[0] == encoded
```

Add tests for:

- bare `tl.constexpr` keeping descriptor + 64-bit payload;
- typed int8/int16/int32/int64/uint widths using 1/2/4/8 bytes;
- a type list reserving its largest width and selecting the smallest fitting type, then canonical name;
- exact dynamic `None` using no key field but remaining in the public call;
- baked `Constexpr(value=...)` leaving neither public call argument nor key field;
- exact binary64 bits distinguishing `-0.0` from `0.0`;
- bool accepted only by `tl.uint1`;
- fp32 and every lower-precision float rejected for constexpr;
- invalid non-power values rejected only in checked dynamic mode, while baked invalid values always fail at creation.

- [ ] **Step 2: Confirm constexpr tests fail**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_launcher.py -k 'constexpr or power_of_two' -q
```

Expected: new typed-width and compact-encoding assertions FAIL.

- [ ] **Step 3: Implement typed constexpr packing**

Replace the old macro with:

```c
static INTJ_ALWAYS_INLINE int
intj_decode_constexpr(PyObject *o, const char *pname, intj_decoded *out);
```

The helper performs structural decoding only. Jinja uses canonical type data to:

- validate exact representability when checked;
- choose a type-list member by smallest width then canonical name;
- preserve the original Python object for the compile callback;
- write the selected descriptor if needed;
- write one little-endian payload of the reserved width.

Never round Python float to fp32. Typed float constexpr accepts only fp64.

- [ ] **Step 4: Implement the compact signed-power encoding**

Add an always-inline helper:

```c
static INTJ_ALWAYS_INLINE uint8_t intj_power_of_two_or_zero(int64_t value) {
  if (value == 0)
    return 0;
  uint64_t magnitude =
      value < 0 ? (uint64_t)(-(__int128)value) : (uint64_t)value;
  unsigned exponent = (unsigned)__builtin_ctzll(magnitude);
  uint8_t code = (uint8_t)(exponent + 1);
  return value < 0 ? (uint8_t)(0x80u | code) : code;
}
```

The checked predicate is `value == 0 || (magnitude & (magnitude - 1)) == 0` and rejects Python bool. Unsigned-only type lists reject negative values.

- [ ] **Step 5: Verify and commit**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_launcher.py -k 'constexpr or power_of_two or rendered_key' -q
pyright
git add intj/launcher.py intj/runtime/intj_runtime.h intj/runtime/entry.c.jinja tests/test_launcher.py
git commit -m "Pack typed constexpr annotations"
```

Expected: focused tests PASS and pyright reports `0 errors`.

---

### Task 8: Compile the final annotated ASTSource

**Files:**
- Modify: `intj/launcher.py:329-358,674-765`
- Modify: `tests/test_annotation.py`
- Modify: `tests/test_launcher.py:300-393`

**Interfaces:**
- Produces: frozen `CompilerInput` for invariant comparison
- Produces: `_compiler_input(jit_func, params, public_args, backend) -> CompilerInput`
- Produces: `_triton_specialize(...)`, a thin wrapper over Triton's `native_specialize_impl`
- Changes: compile callback calls Triton's `ASTSource(...)` and `compile(...)` directly

- [ ] **Step 1: Write compiler-input unit tests**

Add a pure helper test using a fake specialization callback:

```python
def test_compiler_input_applies_fixed_and_automatic_fields(monkeypatch, tmp_path):
    from intj.annotation import DeviceBinding, _resolve_annotations
    from intj.launcher import _compiler_input, _render_params

    class BackendStub:
        @staticmethod
        def parse_attr(desc):
            attrs = []
            if "D" in desc:
                attrs.append(["tt.divisibility", 16])
            if "S" in desc:
                attrs.append(["tt.pointer_range", 32])
            return attrs

    kernel = kernel_from_source(
        tmp_path,
        "import triton\\n\\n@triton.jit\\ndef kernel(x, n, BLOCK):\\n    pass\\n",
    )
    resolved = _resolve_annotations(
        kernel,
        {
            "x": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=Assume(Aligned(16), PointerRange(32)),
            ),
            "n": Argument(type=tl.int32, specialize=NEVER),
            "BLOCK": Constexpr(type=tl.int32),
        },
    )
    params, _, _ = _render_params(resolved, DeviceBinding.NOT_FIXED)
    monkeypatch.setattr(
        "intj.launcher._triton_specialize",
        lambda value, **kwargs: ("i32", ""),
    )
    source = _compiler_input(kernel, params, (object(), 7, 128), BackendStub())
    assert source.signature == (("x", "*fp32"), ("n", "i32"), ("BLOCK", "constexpr"))
    assert source.constants == (((2,), ("int", "128")),)
    assert source.attrs == (((0,), (("tt.divisibility", 16), ("tt.pointer_range", 32))),)
```

Add cases for AUTO equality-to-one becoming a constexpr, NEVER keeping integer `1` as a runtime slot, typed pointer `None` staying a pointer, baked values being inserted in declaration order, and type-list selection using Triton's inferred type.

- [ ] **Step 2: Confirm compiler-input tests fail**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_annotation.py -k compiler_input -q
```

Expected: FAIL because `CompilerInput` and `_compiler_input()` do not exist.

- [ ] **Step 3: Build an immutable compiler input**

Add:

```python
@dataclasses.dataclass(frozen=True)
class CompilerInput:
    signature: tuple[tuple[str, str], ...]
    constants: tuple[tuple[tuple[int, ...], tuple[object, ...]], ...]
    attrs: tuple[tuple[tuple[int, ...], tuple[tuple[str, int], ...]], ...]
    values: tuple[tuple[tuple[int, ...], object], ...] = dataclasses.field(
        compare=False, hash=False, repr=False
    )

    def ast_source(self, jit_func: JitFunction) -> Any:
        from triton.compiler import ASTSource

        return ASTSource(
            jit_func,
            dict(self.signature),
            dict(self.values),
            {path: [list(attr) for attr in attrs] for path, attrs in self.attrs},
        )
```

Implement `_triton_specialize` as a direct call to Triton's
`native_specialize_impl`, the primitive used by its generated binder. Call it
only for a public runtime parameter whose type or applicable fact remains AUTO,
or whose allowlist needs inferred-type selection. Do not invoke it for baked or
bound fixed parameters. Compare tagged `constants` in the invariant validator;
`values` exists only to pass the original Python value to `ASTSource`. Apply
canonical overrides after Triton's result:

- assumed `EqualTo(1)` -> signature `constexpr` and constant `1`;
- typed/bare constexpr -> signature `constexpr` and original Python constant;
- assumed alignment/range -> parsed attrs;
- NEVER -> no attr;
- fixed ordinary type -> canonical signature spelling.

- [ ] **Step 4: Compile the ASTSource without touching JITFunction state**

Replace `jit_func.warmup(...)` in the callback with:

```python
compiler_input = _compiler_input(jit_func, params, args, backend)
src = compiler_input.ast_source(jit_func)
from triton.compiler import compile as triton_compile

kernel = triton_compile(
    src,
    target=target,
    options=canonical_options.__dict__,
)
kernel._init_handles()
```

Keep the existing metadata refusals and kernel lifetime list. Do not call
`warmup()` or `_do_compile()`: both route through mutable
`jit_func.device_caches`, while the compiler's own disk cache already keys the
annotated `ASTSource`.

Change the cold-path invariant map to:

```python
seen: dict[bytes, CompilerInput] = {}
previous = seen.setdefault(keyblob, compiler_input)
if previous != compiler_input:
    raise RuntimeError(
        "intj: one spec key maps to two annotated ASTSource inputs; this is an intj bug"
    )
```

For a zero-word no-map input, perform the same comparison against one saved `CompilerInput`.

- [ ] **Step 5: Add AMD runtime differential tests**

Add annotated kernels that compare Triton output with intj for:

- forced scalar `i32`/`fp64` and exact tensor pointer types;
- AUTO, NEVER, and multiple assumed facts;
- null pointer from explicit pointer `None` and integer `0`;
- baked ordinary values and baked constexpr values;
- typed constexpr widths and positive/negative power encodings.

Extend `test_spec_key_is_never_coarser_than_triton` so its oracle is the final `CompilerInput`, not raw binder output. Mutate each new descriptor/fact contribution locally while writing the tests and confirm at least one targeted case fails before restoring it.

- [ ] **Step 6: Verify and commit**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_annotation.py tests/test_launcher.py -k 'compiler_input or annotated or spec_key' -q
pyright
git add intj/launcher.py tests/test_annotation.py tests/test_launcher.py
git commit -m "Compile canonical annotated AST sources"
```

Expected: focused tests PASS and pyright reports `0 errors`.

---

### Task 9: Bind tensor and pointer arguments into vectorcall handles

**Files:**
- Modify: `intj/launcher.py`
- Modify: `intj/runtime/intj_runtime.h`
- Modify: `intj/runtime/entry.c.jinja`
- Test: `tests/test_launcher.py`

**Interfaces:**
- Produces: private Python `LauncherFactory.bind(**values)`
- Produces: native heap `BoundLauncher` with `tp_vectorcall_offset`
- Produces: module method `make_bound(signature, *values)`

- [ ] **Step 1: Write binding API and ownership tests**

Add:

```python
def test_bind_requires_every_bound_name_once(bound_kernel):
    factory = make_launcher(
        bound_kernel,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.TENSOR,
            ),
            "p": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.POINTER,
            ),
        },
        no_gpu=True,
    )
    with pytest.raises(TypeError, match="not callable"):
        factory(0, 0, 1, 4)
    with pytest.raises(TypeError, match="missing.*p"):
        factory.bind(x=torch.ones(4))
    with pytest.raises(TypeError, match="unknown.*other"):
        factory.bind(x=torch.ones(4), p=0, other=1)


def test_bound_launcher_is_vectorcall_and_hides_bound_parameters(bound_kernel):
    factory = make_launcher(
        bound_kernel,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.TENSOR,
            ),
            "p": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=BindValue.POINTER,
            ),
        },
        no_gpu=True,
    )
    launch = factory.bind(x=torch.ones(4), p=0)
    assert tuple(inspect.signature(launch).parameters) == (
        "device", "stream", "grid", "n"
    )
    assert type(launch).__flags__ & (1 << 11)  # Py_TPFLAGS_HAVE_VECTORCALL
    assert launch(0, 0, 1, 4) is None
```

Add tests that:

- every `bind()` returns a distinct callable retaining its extension;
- actual bound values do not change module identity;
- omitted tensor type is inferred at bind and different inferred dtypes select different module identities;
- omitted tensor type plus bound `None` materializes the same constexpr-`None` compiler input as Triton's unannotated `None`;
- TENSOR accepts tensor or `None`, rejects integer zero, and rereads pointer/storage metadata after `tensor.set_()`;
- POINTER accepts tensor, exact int, `data_ptr()` object, or `None` and snapshots the address;
- pointer zero and `None` are null pointers;
- pointer binding with `PointerRange(32)` emits one creation-time `RuntimeWarning` in checked mode and does not try to verify range;
- bind-time semantic checks disappear when verification is disabled, but structural memory-safety checks remain.

- [ ] **Step 2: Confirm binding tests fail**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_launcher.py -k bind -q
```

Expected: FAIL because `make_launcher()` still returns only a directly callable C function.

- [ ] **Step 3: Add the minimal Python factory**

Add a private frozen factory in `launcher.py`:

```python
@dataclasses.dataclass(frozen=True)
class LauncherFactory:
    jit_func: JitFunction
    params: tuple[ResolvedParam, ...]
    options: tuple[tuple[str, Any], ...]
    torch_access: TorchAccess
    kernel_cache: KernelCache
    verify_annotation: bool
    no_gpu: bool
    bind_device_requested: bool

    def bind(self, **values: object) -> Callable[..., None]:
        if self.bind_device_requested:
            raise TypeError("intj: use bind_device(device_ordinal, **values)")
        return self._bind(None, values)

    def bind_device(self, *args: object, **values: object) -> Callable[..., None]:
        if len(args) != 1:
            raise TypeError("intj: bind_device requires exactly one positional device ordinal")
        return self._bind(args[0], values)
```

`_bind()` requires exactly the canonical bound names, resolves missing TENSOR types, materializes the effective `RenderContext`, loads the module, builds an `inspect.Signature` with bound parameters removed, and calls the module's native `make_bound` method. It passes values in declaration order, never by a generated C identifier.

Return `LauncherFactory` whenever any parameter has a binding mode. Without `bind_device=True`, `bind_device()` raises; with it, `bind()` raises.

- [ ] **Step 4: Add one reusable native call core**

Refactor the body of `entry` into a static `intj_call(...)` that receives:

```c
static PyObject *intj_call(
    intj_state *st,
    intj_cache *cache,
    intj_kernel **fixed_kernel,
    int fixed_device,
    int64_t device_ordinal,
    uint64_t stream,
    PyObject *const *args,
    Py_ssize_t nargs,
    intj_bound_launcher *bound);
```

The existing module `entry` parses dynamic device/stream/grid and calls this core with module cache state. The bound vectorcall function parses its shorter public shape, obtains saved bound values from its object, and calls the same core.

- [ ] **Step 5: Implement native bound ownership**

Create the heap type with `PyType_FromModuleAndSpec`, set `Py_TPFLAGS_HAVE_VECTORCALL`, and store:

```c
typedef struct intj_bound_launcher {
  PyObject_HEAD
  vectorcallfunc vectorcall;
  PyObject *module;
  PyObject *signature;
  PyObject *owners[INTJ_NBOUND];
  uint64_t pointer_bits[INTJ_NBOUND];
  intj_cache cache;
  int cache_ready;
  intj_kernel *fixed_kernel;
  int fixed_device;
  int64_t device_ordinal;
  int32_t device_handle;
#if defined(INTJ_ACCESS_CXX)
  at::Tensor *tensors[INTJ_NBOUND];
#endif
} intj_bound_launcher;
```

Use at least one physical array element when `INTJ_NBOUND == 0`. For TENSOR under CXX, allocate an `at::Tensor` copy with `new (std::nothrow)` and delete it in `tp_dealloc`; do not retain the Python wrapper. Under SHIM/CPYTHON retain a strong `PyObject *` and reread it every launch. For POINTER, snapshot the address at bind and retain a strong owner when the source is not an exact integer.

Catch every C++ exception before returning to CPython. Traverse and clear the module, signature, and Python owners. Expose read-only `__signature__` and `__self__` getters.

- [ ] **Step 6: Verify and commit**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_launcher.py -k bind -q
pyright
git add intj/launcher.py intj/runtime/intj_runtime.h intj/runtime/entry.c.jinja tests/test_launcher.py
git commit -m "Bind arguments into vectorcall launchers"
```

Expected: binding tests PASS and pyright reports `0 errors`.

---

### Task 10: Bind devices and bypass the map when no key remains

**Files:**
- Modify: `intj/launcher.py:484-555`
- Modify: `intj/runtime/entry.c.jinja`
- Modify: `tests/test_launcher.py`

**Interfaces:**
- Adds: `make_launcher(..., bind_device: bool = False)`
- Adds backend data: `Backend.device_symbol`
- CUDA value: `cuDeviceGet`
- HIP value: `hipDeviceGet`
- Fixed-device native handles own their cache or one nullable kernel pointer

- [ ] **Step 1: Write fixed-device API and cache tests**

Add:

```python
def test_bind_device_requires_one_positional_ordinal(device_kernel):
    factory = make_launcher(device_kernel, bind_device=True, no_gpu=True)
    with pytest.raises(TypeError, match="exactly one positional"):
        factory.bind_device()
    with pytest.raises(TypeError, match="exactly one positional"):
        factory.bind_device(0, 1)
    launch = factory.bind_device(0)
    assert tuple(inspect.signature(launch).parameters) == ("stream", "grid", "x")


def test_fixed_device_handles_own_independent_caches(device_kernel):
    factory = make_launcher(device_kernel, bind_device=True, no_gpu=True)
    first = factory.bind_device(0)
    second = factory.bind_device(0)
    calls = []

    def compile_once_per_handle(key, nparams, device, *args):
        calls.append((bytes(key), device))
        return 0, 1, 0, nparams

    first.__self__.set_compile_callback(compile_once_per_handle)
    first(0, 1, 7)
    first(0, 1, 7)
    second(0, 1, 7)
    assert len(calls) == 2
```

Add a no-map case with fixed type and NEVER specialization. Inspect its rendered source and assert:

```python
launch = factory.bind_device(0)
source_path = pathlib.Path(launch.__self__.__file__)
cpp_path = source_path.with_suffix(".cpp")
source = (cpp_path if cpp_path.exists() else source_path.with_suffix(".c")).read_text()
call_body = source[
    source.index("static PyObject *intj_call"):
    source.index("static PyObject *entry")
]
assert "#define INTJ_HAS_KEY 0" in source
assert "intj_hash(" not in call_body
assert "intj_cache_get(" not in call_body
```

Call one handle twice and assert its replacement compile callback runs once. Add a case with one dynamic constexpr proving a fixed-device handle still uses its own map.

- [ ] **Step 2: Confirm fixed-device tests fail**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_launcher.py -k 'bind_device or no_map or independent_caches' -q
```

Expected: FAIL because device binding and per-handle cache state do not exist.

- [ ] **Step 3: Put device lookup on the backend**

Add `device_symbol: str` to `Backend` and set:

```python
class HipBackend(Backend):
    device_symbol = "hipDeviceGet"


class CudaBackend(Backend):
    device_symbol = "cuDeviceGet"
```

Add `device_symbol` to `RenderContext`. The template resolves it beside the launch symbol and uses this common signature:

```c
typedef int32_t (*intj_device_get_t)(int32_t *device, int32_t ordinal);
```

`bind_device()` validates an exact Python int in the C `int32_t` non-negative range, calls the backend resolver without changing the current device, and stores both ordinal and returned handle. In host mode, store the ordinal as the synthetic handle and do not resolve a driver.

- [ ] **Step 4: Materialize fixed-device factories**

If `bind_device=True`, always return `LauncherFactory` even when there are no bound kernel arguments. `bind_device(*args, **kwargs)` accepts exactly one positional ordinal and requires all bound kernel names through keywords. The returned vectorcall signature starts with `stream, grid`; dynamic-device launchers keep `device, stream, grid`.

Before a real first compilation, compare the saved ordinal with `_current_device()` exactly as the current callback does. The saved driver handle is instance state and never enters `RenderContext` or `ModuleKey`.

- [ ] **Step 5: Select per-handle map or nullable-pointer path**

For fixed-device handles:

- if `context.nwords > 0`, initialize and use `bound->cache`;
- if `context.nwords == 0`, leave the cache uninitialized and use `bound->fixed_kernel`;
- first no-map call invokes the callback and installs one record;
- later calls use one `INTJ_UNLIKELY(!bound->fixed_kernel)` branch and no hash/map operation;
- deallocation frees the selected state exactly once.

Dynamic-device and non-bound launchers continue to use module state, with the device byte at `context.device_offset`.

- [ ] **Step 6: Compile-check both backend renderings**

Add a test that renders and builds one host module with `error_style="return"`/`hipDeviceGet` and one with `error_style="outparam"`/`cuDeviceGet`. It must compile both generated branches without loading a CUDA driver.

- [ ] **Step 7: Verify and commit**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_launcher.py -k 'bind_device or no_map or cuda_template' -q
pyright
git add intj/launcher.py intj/runtime/entry.c.jinja tests/test_launcher.py
git commit -m "Bind devices and skip empty kernel maps"
```

Expected: fixed-device tests PASS, both backend variants compile, and pyright reports `0 errors`.

---

### Task 11: Complete checked binding and runtime invariant coverage

**Files:**
- Modify: `intj/launcher.py`
- Modify: `intj/runtime/intj_runtime.h`
- Modify: `intj/runtime/entry.c.jinja`
- Modify: `tests/test_launcher.py`

**Interfaces:**
- Keeps verification at the point where each value can change
- Keeps `BindValue.POINTER + PointerRange(32)` trusted and warning-only
- Extends the invariant validator to module maps, per-handle maps, and no-map handles

- [ ] **Step 1: Add the remaining verification matrix**

Add parameterized host-only and AMD cases for:

```python
@pytest.mark.parametrize(
    "mode,value,valid",
    [
        (BindValue.TENSOR, None, True),
        (BindValue.TENSOR, 0, False),
        (BindValue.POINTER, None, True),
        (BindValue.POINTER, 0, True),
        (BindValue.POINTER, 4096, True),
    ],
)
def test_binding_null_and_structural_rules(mode, value, valid, pointer_kernel):
    factory = make_launcher(
        pointer_kernel,
        extra_annotation={
            "x": Argument(
                type=tl.pointer_type(tl.float32),
                specialize=NEVER,
                bind_value=mode,
            )
        },
        verify_annotation=True,
        no_gpu=True,
    )
    if valid:
        factory.bind(x=value)
    else:
        with pytest.raises(TypeError, match="x"):
            factory.bind(x=value)
```

Also cover:

- bound TENSOR alignment and pointer-range checks rerun after `set_()` changes storage;
- bound POINTER alignment checks happen once at bind;
- bound POINTER range assumption warns once at factory creation and is not checked;
- unchecked binding omits semantic checks but rejects unsafe object layouts;
- errors name the parameter and failed constraint before cache lookup;
- each generated checked argument has one predicted failure branch followed by one assume;
- unchecked source contains no semantic verification expression.

- [ ] **Step 2: Add invariant cases for all cache ownership modes**

For module cache, fixed-device per-handle map, and fixed-device no-map:

1. install a recording compile callback;
2. call with pairs that should share compiler input and assert one miss;
3. call with pairs that differ in final signature, constants, or attrs and assert distinct misses;
4. deliberately remove each relevant descriptor or payload in a local test mutation and confirm the assertion fails before restoring it.

For the no-map path, pass two accepted calls and assert the saved `CompilerInput` comparison remains equal.

- [ ] **Step 3: Keep warning and check behavior out of the hot path**

Emit the unverified `PointerRange(32)` warning in `make_launcher()` once per affected parameter:

```python
warnings.warn(
    f"intj: pointer-range assumption for bound pointer {param.name!r} "
    "cannot be verified from an address and will be trusted",
    RuntimeWarning,
    stacklevel=2,
)
```

Do not emit a bind-time or launch-time range branch for that case. All other checked constraints use already-decoded values and no extra Python calls.

- [ ] **Step 4: Run the complete behavioral suite**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_annotation.py tests/test_launcher.py -q
pyright
```

Expected: all annotation, host-only, AMD, generated-source, and invariant tests PASS; pyright reports `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add intj/launcher.py intj/runtime/intj_runtime.h intj/runtime/entry.c.jinja tests/test_launcher.py
git commit -m "Verify annotated launch contracts"
```

---

### Task 12: Benchmark and document the feature

**Files:**
- Modify: `benchmarks/bench_launch.py`
- Modify: `docs/Usage.md`
- Modify: `tests/test_launcher.py`

**Interfaces:**
- Adds CLI: `python benchmarks/bench_launch.py [--no-gpu] [--iters N] [--batches N]`
- Reports repeated-batch medians, absolute time, matching-baseline delta, and construction/binding time
- Adds no performance threshold to pytest

- [ ] **Step 1: Write a no-GPU benchmark smoke test**

Add:

```python
def test_benchmark_matrix_runs_without_gpu():
    run = subprocess.run(
        [
            sys.executable,
            "benchmarks/bench_launch.py",
            "--no-gpu",
            "--iters",
            "20",
            "--batches",
            "3",
        ],
        cwd=pathlib.Path(__file__).parents[1],
        env={**os.environ, "PYTHONPATH": str(pathlib.Path(__file__).parents[1])},
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    for label in (
        "auto map",
        "reduced key",
        "verify off",
        "verify on",
        "baked",
        "bound tensor",
        "bound pointer",
        "fixed device map",
        "fixed device no-map",
    ):
        assert label in run.stdout
```

- [ ] **Step 2: Confirm the smoke test fails**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests/test_launcher.py -k benchmark_matrix -q
```

Expected: FAIL because the benchmark has no argument parser, host mode, or matrix.

- [ ] **Step 3: Replace the one-shot timer with repeated medians**

Use stdlib only:

```python
import argparse
import statistics


def bench(fn, iters, batches, sync):
    for _ in range(100):
        fn()
    sync()
    samples = []
    for _ in range(batches):
        start = time.perf_counter_ns()
        for _ in range(iters):
            fn()
        sync()
        samples.append((time.perf_counter_ns() - start) / iters)
    return statistics.median(samples)
```

Use `sync=lambda: None` under `--no-gpu` and `torch.cuda.synchronize` otherwise. Use CPU tensors in host mode and GPU tensors in normal mode.

- [ ] **Step 4: Add the targeted matrix**

Construct and time these rows against their matching AUTO baseline:

1. current AUTO map path;
2. fixed type/facts with a reduced key;
3. identical constraints with verification off and on;
4. baked ordinary plus constexpr values;
5. bound tensor vectorcall;
6. bound pointer vectorcall;
7. fixed-device handle with a remaining dynamic key;
8. fixed-device no-map handle.

Warm each launcher before measurement. Print construction time and, where applicable, binding time separately. Report nanoseconds per call, absolute delta, and percentage delta. Do not assert a timing limit.

- [ ] **Step 5: Rewrite Usage documentation**

Update `docs/Usage.md` with:

- all new `make_launcher` keyword arguments and defaults;
- public annotation constructors, shorthands, presets, merge order, AUTO/NEVER/Assume semantics, and exact `None` behavior;
- supported ordinary/constexpr dtypes and power-of-two-or-zero encoding;
- baked values, `bind()`, `bind_device()`, callable signatures, ownership, and null behavior;
- checked versus unchecked guarantees and the pointer-range warning;
- descending-alignment key layout, fixed-device cache ownership, and no-map path;
- host-only benchmark scope and both benchmark commands;
- explicit statement that CUDA remains compile-checked but runtime-untested.

Remove the old statements that `extra_annotation` is reserved and that every declared parameter is always passed.

- [ ] **Step 6: Run final verification**

Run:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python -m pytest tests -q
pyright
PYTHONPATH=$PWD /tmp/gb2/bin/python benchmarks/bench_launch.py --no-gpu --iters 20000 --batches 7
PYTHONPATH=$PWD /tmp/gb2/bin/python benchmarks/bench_launch.py --iters 20000 --batches 7
git diff --check
```

Expected:

- pytest PASS, with only environment-dependent documented skips;
- pyright reports `0 errors`;
- both benchmark modes print all matrix rows and medians;
- `git diff --check` prints nothing.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/bench_launch.py docs/Usage.md tests/test_launcher.py
git commit -m "Benchmark and document argument annotations"
```

---

## Final Review Checklist

- [ ] Search the plan and implementation for placeholders:

```bash
rg -n 'T[B]D|implement l[a]ter|fill i[n]|similar to T[a]sk|appropriate e[rror]' \
  docs/superpowers/plans/2026-09-23-argument-annotations.md \
  intj tests benchmarks docs/Usage.md
```

Expected: no implementation placeholders. The documented future untyped-pointer dtype input remains a stated non-goal, not a code marker.

- [ ] Confirm the public names and private sentinels:

```bash
PYTHONPATH=$PWD /tmp/gb2/bin/python - <<'PY'
import intj
assert "UNSET" not in intj.__all__
for name in (
    "Annotation", "Argument", "Constexpr", "Specialization", "AUTO", "NEVER",
    "Assume", "Fact", "EqualTo", "Aligned", "PointerRange", "BindValue",
    "INT_TYPES", "FLOAT_TYPES", "__version__",
):
    assert name in intj.__all__, name
PY
```

- [ ] Confirm the worktree contains no accidental changes:

```bash
git status --short
git diff --stat HEAD
```

Expected: only intentional implementation files plus the user's pre-existing `.gitignore` modification.

---
