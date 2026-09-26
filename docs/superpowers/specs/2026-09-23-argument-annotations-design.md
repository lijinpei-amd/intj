# Argument annotations

## Problem

`intj.make_launcher()` currently accepts only Triton's existing unannotated
arguments and bare `tl.constexpr` parameters. Its `extra_annotation` argument is
reserved and rejected.

That leaves useful information unavailable to the generated launcher:

- The exact type or allowed type set of an argument.
- Whether Triton's value, alignment, and pointer-range specialization is
  automatic, disabled, or guaranteed by the caller.
- The storage width of a constexpr value.
- Values and tensor/pointer handles that should be removed from the public launch
  call and supplied by the launcher.
- A device that should be selected once rather than included in every cache key.

The annotation must affect compilation and module identity. A generated launcher
may omit a value or fact from its per-launch key only when the final compiler
input cannot vary for that omission.

The safety invariant remains:

> same intj kernel-cache key implies the same annotated Triton specialization

Here, "Triton specialization" means the `ASTSource` signature, constexpr values,
and specialization attributes that intj actually compiles. Bound launchers may
own separate caches; the invariant applies within the cache used by one launch.

## Goals

- Add an intj-owned annotation hierarchy for normal and constexpr arguments.
- Accept annotations inline and through `extra_annotation`, with deterministic
  field-by-field merging.
- Keep canonical annotations immutable, hashable, and JSON-serializable.
- Let annotations change the signature, constexprs, and specialization facts
  presented to Triton's compiler without mutating the original `JITFunction`.
- Remove fixed data from the per-launch key and remove baked or bound parameters
  from the public launch call.
- Eliminate the hash map entirely when a fixed-device launcher has no dynamic
  specialization input.
- Keep the checked and unchecked hot paths small and benchmark their cost without
  requiring a GPU.

## Non-goals

- Supporting arbitrary Python annotation or fact subclasses.
- Casting or reinterpreting tensor storage.
- Adding scalar conversion for fp8, fp16, or bf16.
- Inferring a pointee type from a raw pointer or arbitrary `data_ptr()` object.
- Mutating or cloning the original `JITFunction`.
- Supporting keyword launch arguments, variadic kernel parameters, dynamic
  compile options, or other currently refused kernel shapes.
- Making `no_gpu=True` execute or validate GPU code.

## Public API

The public definitions live in `intj/annotation.py` and are re-exported from
`intj`:

```python
import enum
from dataclasses import dataclass
from typing import Sequence

import triton.language as tl


@dataclass(frozen=True)
class Annotation: ...


@dataclass(frozen=True)
class Specialization: ...


@dataclass(frozen=True)
class Auto(Specialization): ...


@dataclass(frozen=True)
class Fact: ...


@dataclass(frozen=True)
class _Unset: ...


UNSET = _Unset()


class BindValue(enum.Enum):
    TENSOR = "tensor"
    POINTER = "pointer"


@dataclass(frozen=True)
class Argument(Annotation):
    type: tl.dtype | Sequence[tl.dtype | None] | None | Auto | _Unset = UNSET
    specialize: Specialization | _Unset = UNSET
    value: object = UNSET
    bind_value: BindValue | None = None


@dataclass(frozen=True)
class Constexpr(Annotation):
    type: tl.dtype | Sequence[tl.dtype | None] | None | Auto | _Unset = UNSET
    power_of_two_or_zero: bool = False
    value: object = UNSET
```

The snippet uses the private `_Unset` type and its `UNSET` singleton to explain
the constructor state. Neither is exported. `AUTO` is the only public type-field
value for requesting Triton's normal inference.

`Sequence` inputs are normalized to tuples during construction. Public
annotation objects remain immutable and hashable even when the caller passes a
list.

The specialization values are:

```python
AUTO
NEVER
Assume(EqualTo(1), Aligned(16), PointerRange(32))
```

`AUTO` and `NEVER` are immutable singleton `Specialization` values. `Assume` is
an immutable `Specialization` containing a canonical tuple of intj-owned `Fact`
instances. `EqualTo`, `Aligned`, and `PointerRange` are immutable `Fact`
subclasses.

`Fact` is a public taxonomy and typing base, not an extension point. Validation
accepts only the exact built-in `EqualTo`, `Aligned`, and `PointerRange` classes;
instances of `Fact` itself and external subclasses are rejected.

Version one accepts only the specialization facts Triton 3.8 exposes to this
launcher:

- `EqualTo(1)`
- `Aligned(16)`
- `PointerRange(32)`

Other fact values are rejected rather than accepted with no compiler effect.

Frequently used type sets are immutable tuples:

```python
INT_TYPES = (tl.int32, tl.int64, tl.uint64)
FLOAT_TYPES = (tl.float32,)
```

There is no `UINT_TYPES` preset. `tl.uint1` is boolean-like and remains an
explicit type rather than a member of `INT_TYPES`.

Existing values are accepted as shorthands:

```python
tl.int32      # Argument(type=tl.int32)
tl.constexpr  # Constexpr()
None          # Argument(type=None)
```

The launcher adds these keyword arguments:

```python
make_launcher(
    jit_func,
    extra_annotation=None,
    bind_device=False,
    verify_annotation=False,
    no_gpu=False,
    ...,
)
```

Examples:

```python
@triton.jit
def kernel(
    x: Argument(
        type=tl.pointer_type(tl.float32),
        specialize=Assume(Aligned(16), PointerRange(32)),
        bind_value=BindValue.TENSOR,
    ),
    n: Argument(type=tl.int32, specialize=NEVER),
    scale: Argument(type=tl.float32, specialize=NEVER, value=2.0),
    BLOCK: Constexpr(type=tl.int32, power_of_two_or_zero=True),
    WARPS: Constexpr(type=tl.int32, value=4),
):
    ...


factory = intj.make_launcher(kernel, bind_device=True)
launch = factory.bind_device(0, x=tensor)
launch(stream, grid, n, BLOCK)
```

`scale`, `WARPS`, the device, and `x` are absent from the final callable. `n` and
`BLOCK` remain in declaration order after `stream` and `grid`.

## Accepted annotation sources and merging

Each declared parameter may receive fields from:

1. Its inline Python annotation.
2. `extra_annotation[name]` passed to `make_launcher()`.
3. Triton's existing `do_not_specialize` and
   `do_not_specialize_on_alignment` decorator state.

Inline and extra values accept only:

- An exact intj `Annotation` instance.
- A Triton `tl.dtype`, as shorthand for `Argument(type=...)`.
- Bare `tl.constexpr`, as shorthand for `Constexpr()`.
- Bare `None`, as shorthand for `Argument(type=None)`.

`extra_annotation` is a mapping from declared parameter name to one of those
values. Unknown names are rejected.

Fields merge independently. A private `UNSET` means the source omitted that
field and another source may fill it. Equal duplicate fields are accepted.
Unequal duplicate fields, incompatible facts, and disagreement with Triton's
decorator flags are rejected.

Explicit `AUTO` is specified, not omitted. It conflicts with another source
that fixes the same field. After all sources merge, every remaining `UNSET`
canonicalizes to `AUTO`.

`None` is never an omission marker:

- In `type`, it is the exact `None` type.
- In a type sequence, it is one allowed type.
- In `value`, it is an explicitly baked value.
- As a bare annotation source, it means `Argument(type=None)`.

Type and value are independent. `Argument(type=None)` and
`Constexpr(type=None)` still leave a parameter in the public launch call;
adding `value=None` removes it.

The merged result, not either input object, is canonicalized and hashed.

## Canonical representation and module identity

Each rendered `Param` stores one immutable canonical annotation containing:

- The kind: `Argument` or `Constexpr`.
- `AUTO`, exact `None`, or a tuple of canonical Triton type names.
- The mode of every supported specialization fact: `AUTO`, `NEVER`, or
  `ASSUME`.
- The canonical assumed facts.
- The `power_of_two_or_zero` flag.
- The binding mode: `None`, `TENSOR`, or `POINTER`.
- One canonical baked-value tuple.
- The computed key-field width and byte offset, when the parameter contributes
  dynamic key state.

Type and fact collections are deduplicated and sorted by stable canonical names.
Ordering in caller-provided lists therefore has no semantic or hashing effect.

The private constructor sentinel never enters the canonical value. Baked values
use one tagged tuple:

```text
()                                  no baked value
("none",)                           baked None
("bool", false)                    baked bool
("int", "128")                     baked integer
("float64", "8000000000000000")   exact IEEE-754 bits
```

The tags keep Python equality aliases such as `False == 0` and `-0.0 == 0.0`
from making canonical annotations compare or hash equal. The representation is
immutable and JSON-serializable.

Actual `BindValue` objects and actual fixed-device handles are instance state.
They do not enter `RenderContext` or `ModuleKey`. Their binding mode does enter
the context, and a type inferred during binding becomes the exact type in the
effective context loaded for that bound callable.

`RenderContext.params` contains the canonical annotations. Because `ModuleKey`
serializes and hashes the complete context, no second raw-annotation field is
added. `ModuleKey.params` must not retain a raw, unmerged representation.

For a fixed JIT function and source, equivalent merged annotations share a
rendered module. Canonically different structural annotations produce different
module digests. A baked constexpr or baked ordinary value is structural and
therefore changes the digest; a value supplied later through `bind()` is
instance state and does not.

## Argument type semantics

### Inference, exact types, and allowlists

`Argument()` and `Argument(type=AUTO)` preserve Triton's unspecialized runtime
type inference. The difference exists only while sources merge: an omitted field
may be filled, while explicit `AUTO` conflicts with a fixed type from another
source.

`Argument(type=None)` accepts exactly Python `None`.

A single Triton type forces that signature type. With verification enabled, the
launch path verifies the Python value kind and exact representable range before
packing it.

A type sequence is an allowlist, not a list of coercion candidates. Intj obtains
the unspecialized inferred type and requires that canonical type to be a member
of the list. `None` may be a member. `AUTO` may not be a member. Duplicate
spellings are accepted and removed during canonicalization.

### Supported scalar types

Scalar `Argument.type` accepts:

- All Triton integer dtypes.
- `tl.uint1` for Python `bool`.
- `tl.float32` and `tl.float64` for Python `float`.
- Exact `None`.

Python `bool` is accepted only for `tl.uint1`; it is not treated as an integer.
Integer and floating-point values are not implicitly converted into each other.

Low-precision scalar floating types (`fp8`, `fp16`, and `bf16`) are rejected.
They require rounding and bit-packing machinery outside this feature.

### Pointer types and null values

A tensor pointer type uses Triton's native spelling:

```python
tl.pointer_type(tl.float32)
```

Bare `tl.float32` always means a scalar. A verified runtime tensor must match the
canonical pointer element type exactly. Intj does not cast or reinterpret tensor
storage.

Triton 3.8 treats an integer `0` passed to an unannotated argument as an `i32`
runtime scalar. An explicit pointer annotation changes both integer `0` and
Python `None` into null runtime pointers of that pointer type. The explicit type
wins; those values do not become an `i32` or constexpr-`None` specialization.

Without an explicit pointer type, Python `None` follows Triton's normal behavior
and becomes a constexpr-`None` specialization.

## Ordinary argument key omission and values

An ordinary argument contributes no per-launch key field when:

- Its effective type is one exact type.
- No specialization fact applicable to that type remains `AUTO`.

The argument still remains in the public call and GPU argument array unless it
has `value=...` or a binding mode.

`Argument(value=...)` accepts only non-tensor, non-pointer values. It:

- Resolves one effective type and requires fixed specialization behavior.
- Is validated and converted while creating the launcher.
- Is emitted into the rendered C/C++ extension using its exact bit
  representation.
- Participates in `RenderContext` and `ModuleKey`.
- Is omitted from the public launch call and per-launch key.
- Is supplied as the runtime GPU argument, unless the final Triton
  specialization removes that slot.

`value` and a non-`None` `bind_value` are mutually exclusive.

## Bound tensor and pointer arguments

`BindValue.TENSOR` and `BindValue.POINTER` remove an argument from the callable
returned after binding. They require fixed specialization behavior; no
applicable fact may remain `AUTO`.

`BindValue.TENSOR` accepts a Torch tensor or `None`:

- Under `TorchAccess.CXX`, `bind()` copies and owns an `at::Tensor`. Its
  intrusive reference keeps the `TensorImpl` and storage alive without retaining
  the Python wrapper.
- Under `TorchAccess.SHIM` and `CPYTHON`, `bind()` retains a strong `PyObject*`.
- Each launch rereads the current pointer, size, and required metadata, so
  storage changes remain visible.
- An omitted type is inferred from a tensor during `bind()` and becomes fixed in
  the effective render context.
- Integer values, including zero, are rejected.

`BindValue.POINTER` accepts a Torch tensor, an exact Python integer, an object
with `data_ptr()`, or `None`. It resolves and snapshots the address during
`bind()`. It requires exactly one explicit `tl.pointer_type(...)`; missing,
scalar, or list-valued type annotations raise `ValueError` during
`make_launcher()` because an address supplies no pointee type.

An address alone cannot verify Triton's `PointerRange(32)` fact because it does
not describe the backing allocation size. `BindValue.POINTER` therefore always
trusts that fact, even with `verify_annotation=True`. In checked mode,
`make_launcher()` emits one `RuntimeWarning` identifying the parameter and the
unverifiable fact. The check is not emitted at bind or launch time.

For both binding modes, `None` follows the effective type:

- With an explicit pointer type, it binds a null runtime pointer.
- Without a pointer type, it derives a launcher using Triton's constexpr-`None`
  specialization.

For `BindValue.POINTER`, integer zero is a null runtime pointer because the
required explicit pointer annotation fixes its meaning. It never falls back to
Triton's unannotated `i32` inference.

TODO (future): add a dtype argument to `bind()` if `BindValue.POINTER` needs to
accept an untyped address. Version one deliberately refuses it.

### Binding API and callable ownership

If any parameter has a binding mode, the object returned by `make_launcher()` is
not directly callable. Binding is explicit and keyword-based:

```python
factory = make_launcher(kernel)
launch_a = factory.bind(x=tensor_a, ptr=pointer_a)
launch_b = factory.bind(x=tensor_b, ptr=pointer_b)
```

Every bound parameter name is required exactly once; unknown and duplicate
names are rejected. Each call creates a new native callable with private bound
state.

When `bind_device=True`, `bind()` is not sufficient: the factory must be called
through `bind_device(device_ordinal, **bound_values)` so the device and all bound
kernel arguments are installed in one handle.

The callable is a `METH_FASTCALL` builtin whose `__self__` is a private C heap
`BoundLauncher` object. That object owns the extension and bound values; the
builtin adds no Python frame or argument tuple for positional launches. Its
`__text_signature__` omits bound parameters.

When the device is not bound, bound callable instances use the loaded
extension's normal cache, whose key still includes the runtime device. Actual
bound values do not enter that key because their effective type and
specialization behavior are fixed.

## Constexpr semantics

### Dynamic and baked values

Bare `Constexpr()` and bare `tl.constexpr` preserve the current dynamic-kind,
64-bit-payload behavior. They accept the same `int`, `float`, `bool`, and `None`
values as Triton 3.8.

`Constexpr(value=...)` completely bakes the value:

- The parameter remains in the original Triton function signature.
- It is absent from the generated launcher's call signature.
- It consumes no C/C++ parameter slot or per-launch key storage.
- Intj inserts it into its original position when constructing the annotated
  `ASTSource`.
- Its exact canonical value participates in `RenderContext` and `ModuleKey`.
- Its type and other constraints are checked during `make_launcher()`.

Because the caller supplies no runtime value, `verify_annotation` has no effect
on a baked constexpr.

### Constexpr types and widths

`Constexpr()` and `Constexpr(type=AUTO)` preserve Triton's dynamic value-kind
inference. `Constexpr(type=None)` accepts only a dynamic runtime `None` but does
not remove the parameter; `Constexpr(type=None, value=None)` removes it.

A typed `Constexpr` accepts integer dtypes, `tl.uint1`, `tl.float64`, and exact
`None`. It rejects fp8, fp16, bf16, and fp32. A Python float is binary64;
storing only its rounded fp32 representation while passing the original binary64
value to Triton could map two compiler constants to one intj key.

A single constexpr type requires exact representability. A type list accepts a
value representable by at least one member and reserves a payload as wide as the
largest member. When several members represent the value, selection is
deterministic: smallest width first, then canonical type-name order.

The original Python constexpr value is passed to Triton unchanged. The declared
type controls validation and key storage; it does not introduce lossy constexpr
conversion. Python `bool` is accepted only by `tl.uint1`.

### Signed power of two or zero

`power_of_two_or_zero` is valid only on `Constexpr` and only for integer values.
Python `bool` is rejected. The value must also fit at least one declared integer
type; unsigned-only types reject negative values.

The accepted domain is:

```python
value == 0 or abs(value).bit_count() == 1
```

Dynamic values use one byte and no separate payload:

```text
0       -> 0x00
+2**n   -> n + 1
-2**n   -> 0x80 | (n + 1)
```

This covers `uint64` through `+2**63` and `int64` through `-2**63`.

With verification enabled, invalid dynamic values raise `ValueError`. With
verification disabled, the encoder trusts the contract and invalid values may
alias. Baked values are always checked while creating the launcher.

## Specialization semantics

Specialization is modeled independently for equality-to-one, 16-byte alignment,
and 32-bit pointer range after all sources merge.

`AUTO` uses Triton's normal decision for each launch.

`NEVER` disables every applicable equality, alignment, and pointer-range fact.
Triton's existing decorator flags are folded into individual internal fact
modes. `do_not_specialize` disables equality-to-one and alignment while leaving
pointer range unspecified, matching Triton 3.8.
`do_not_specialize_on_alignment` affects alignment only. Public `NEVER` is
deliberately stronger and disables every applicable fact in the annotated
`ASTSource`.

`NEVER` does not undo inherent constexpr classification. For example, an
untyped runtime `None` still becomes Triton's constexpr-`None` specialization.

`Assume(...)` has four effects for each applicable named fact:

1. With verification enabled, check it at the point where its value can change.
2. Compile with the fact present.
3. Omit the fact from the per-launch map key.
4. Leave unnamed facts available for another source and otherwise `AUTO`.

`Assume(EqualTo(1))` compiles an applicable integer value as a constant and
removes its GPU argument slot. `Aligned(16)` applies to integer scalars and
pointers. `PointerRange(32)` applies to pointers and means the backing storage
fits the backend's signed 32-bit pointer-range specialization.

Fact applicability is conditional:

| Fact | Applicable runtime types |
|---|---|
| `EqualTo(1)` | integer scalars except `tl.uint1` |
| `Aligned(16)` | integer scalars and pointers |
| `PointerRange(32)` | pointers |

`None`, booleans, and floats have no applicable facts. On a mixed type list, a
fact is checked and compiled only for compatible selected types.

Every fact must apply to at least one possible type. Every declared non-`None`
type branch must have at least one value satisfying all facts applicable to that
branch. Impossible branches are rejected rather than silently removed. For a
baked value, only its resolved type remains possible.

Equal duplicate facts are deduplicated. `AUTO` versus `NEVER` or `ASSUME`, and
`NEVER` versus `ASSUME`, are merge conflicts. Distinct compatible assumptions
are combined. Statically impossible combinations, including
`EqualTo(1)` with `Aligned(16)` on an integer branch, raise `ValueError` during
launcher creation.

## Device binding

`make_launcher(bind_device=False)` preserves the current public call shape and
places the validated device ordinal in the per-launch cache key.

`make_launcher(bind_device=True)` returns a non-callable factory. It is bound
with:

```python
launch = factory.bind_device(device_ordinal, x=tensor, ...)
launch(stream, grid, ...)
```

The implementation accepts `bind_device(*args, **kwargs)`, requires exactly one
positional integer device ordinal, and treats all remaining values as
keyword-only bound kernel arguments. This permits any kernel parameter name in
`kwargs`, including names that would collide with a Python method parameter.

The backend supplies the device-resolution symbol as data:

- CUDA: `cuDeviceGet`
- HIP: `hipDeviceGet`

Binding resolves and stores both the ordinal and driver device handle without
changing the current device. The first real compilation still requires that
device to be current, matching the existing launcher's context-safety check.

`RenderContext` stores `DeviceBinding.FIXED` or `DeviceBinding.NOT_FIXED`, not
the actual device. Each `bind_device()` result owns its own kernel-cache state;
handles do not share kernel records even when bound to the same device.

When dynamic specialization remains, the handle owns a normal map keyed only by
those dynamic fields. When none remains, it owns one nullable kernel pointer:

```text
null      first launch compiles and installs the kernel
non-null  launch it directly
```

Compilation remains lazy. The hot no-map path performs one predicted-not-null
branch and no hash or map operation.

## Per-launch key layout

Only fields that can vary within one callable are present. A baked parameter, a
bound parameter with fixed effective type and specialization, and an ordinary
parameter with fixed type and specialization contribute no key bytes. A fixed
device contributes no device byte.

The remaining key is a fixed-size byte array whose size is rounded up to 64 bits
for the existing hash and cache interface. Fields are grouped by descending
alignment to avoid internal padding:

```text
[64-bit constexpr payloads]
[32-bit constexpr payloads]
[16-bit constexpr payloads]
[8-bit descriptors, values, and constexpr payloads]
[dynamic device byte, when present]
[zero padding to 64 bits]
```

Within each width group, parameter-derived fields remain in declaration order.
If one parameter contributes both a descriptor and an 8-bit payload, the
descriptor comes first. The dynamic device byte follows all parameter fields.

An untyped dynamic constexpr receives a kind descriptor and a 64-bit payload. A
typed dynamic constexpr reserves the largest width in its canonical type set.
`power_of_two_or_zero` stores its complete dynamic value in the 8-bit field and
allocates no payload.

Payloads use little-endian bit representations. Signed integers use their
declared-width two's-complement representation. Descriptors distinguish any
runtime type or automatic specialization fact that can change the final
`ASTSource`.

`AUTO` facts participate in the map key. `NEVER` and `ASSUME` facts do not. The
annotation itself is never copied into the per-launch key; its structural form
belongs to `RenderContext` and `ModuleKey`.

If the device is fixed and no parameter contributes a dynamic field, `nwords`
is zero and the generated launcher uses the no-map path described above.

## Annotation verification

`verify_annotation=False` is the default. The canonical boolean is stored in
`RenderContext`, so checked and unchecked launchers have different module
digests and generated source.

With verification disabled:

- No declared dtype, declared-width, allowlist, equality, alignment, or
  pointer-range checks are emitted in launch or binding paths.
- Values outside an annotated width may truncate according to generated ABI
  packing.
- Violating a promise may compile or launch the wrong kernel.
- Checks needed to avoid invalid CPython memory access, represent a value in the
  host storage type, infer a missing type, validate argument counts/grid/stream,
  and report driver errors remain.

The switch is an unsafe performance contract, not permission for generated C to
read an arbitrary `PyObject` layout.

With verification enabled, all checks for one constrained argument share one
fast-path branch:

```cpp
bool valid = /* checks over already-decoded values */;
if (INTJ_UNLIKELY(!valid)) {
    /* Cold path identifies and reports the exact violated constraint. */
    PyErr_Format(...);
    return nullptr;
}
INTJ_ASSUME(valid);
```

`INTJ_UNLIKELY` uses `__builtin_expect` where available. `INTJ_ASSUME` uses the
compiler builtin where available and appears only after the checked branch.

Checks occur where a value can change:

- Ordinary unbound arguments: every launch.
- `BindValue.TENSOR`: every launch, because storage and `data_ptr()` may change.
- `BindValue.POINTER`: supported checks once during `bind()`, because the address
  is snapshotted; `PointerRange(32)` is the warned, unchecked exception described
  above.
- Baked `Argument.value` and `Constexpr.value`: during `make_launcher()`.

Malformed annotations and statically impossible combinations are always
rejected. `verify_annotation` controls runtime promises only.

## Compilation flow

The original `JITFunction` is never mutated, and no replacement `JITFunction`
is constructed. Constructing a duplicate would register another function under
the same module and qualified name and disturb Triton's global state.

On a cache miss:

1. Intj reconstructs the logical argument sequence, inserting baked constexprs,
   baked ordinary values, and bound values at their declared positions.
2. It invokes Triton's binder only where baseline `AUTO` behavior or type-list
   inference is still required.
3. It applies the canonical annotation to produce the final signature,
   constexpr mapping, and specialization attributes.
4. It constructs `ASTSource(jit_func, signature, constexprs, attrs)` and invokes
   Triton's compiler through its existing compilation machinery.
5. It initializes and retains the resulting `CompiledKernel` without mutating
   the `JITFunction`.

The intj map key is not a compiler input. Python receives it only for the
cold-path invariant validator. The required invariant is:

> same intj map key implies the same annotated ASTSource signature, constexprs,
> and attrs

The validator compares against the final annotated compiler input, not the raw
binder result. The no-map path validates that its one effective compiler input
is invariant across every accepted call.

## GPU-free host benchmark mode

`make_launcher(no_gpu=True)` is an explicit benchmark mode. It participates in
`RenderContext` and `ModuleKey` and builds a host-only generated extension.

The host-only path performs:

- Argument decoding and ABI packing.
- Optional annotation verification.
- Key construction and cache lookup, or the no-map nullable-pointer branch.
- Bound `METH_FASTCALL` dispatch.
- Dummy cache-miss installation.

It does not discover an active GPU target, invoke Triton GPU compilation,
resolve a GPU driver symbol, or launch a kernel. `bind_device()` uses a synthetic
device handle in this mode. Calls return normally after the host path completes.
The mode is labeled host-only and makes no end-to-end GPU performance or
correctness claim.

Version one may reject compile options that require a real backend in
`no_gpu=True`; the benchmark uses the default options.

## Errors

No new exception hierarchy is added.

- `TypeError` during annotation construction: a field has the wrong Python
  kind.
- `ValueError` during annotation construction or launcher creation: an empty
  type list, unsupported dtype or fact, source conflict, dead fact, impossible
  type branch, mutually exclusive `value`/`bind_value`, or missing
  `BindValue.POINTER` pointee type.
- `UnsupportedKernel`: a valid annotation is used on a kernel or option shape
  intj does not support.
- `TypeError` during binding or checked launch: the Python value has the wrong
  structural kind.
- `ValueError` or `OverflowError` during checked launch: allowlist rejection,
  failed fact, invalid signed-power-of-two-or-zero value, or numeric range
  failure.

Checked launch-time annotation errors occur before cache lookup and name the
parameter and failed constraint. Unchecked semantic violations have no promised
error behavior.

## Module cache version

`SCHEMA_VERSION` is removed. The package has one version source:

```python
# intj/_version.py
__version__ = "0.1.0"
```

`pyproject.toml` reads it through setuptools dynamic versioning, and
`intj.__version__` re-exports it.

`ModuleKey` stores `intj_version: tuple[int, int]`. Major and minor releases are
cache-changing; patch releases are cache-preserving. Template and runtime-header
content hashes continue to invalidate generated modules when those files change
within a patch-development cycle.

## Verification

### Canonicalization and value types

Tests cover source merging, omitted versus explicit `AUTO`, bare and nested
`None`, reordered and duplicate types/facts, baked-value tags, exact fact class
validation, immutable equality/hash behavior, and JSON serialization.
Equivalent merged annotations produce equal contexts and module digests;
canonically different structural annotations do not.

### Key-field layout

Tests cover every boundary between 64-, 32-, 16-, and 8-bit groups; declaration
order within groups; optional device byte; final padding; largest-member
reservation for type lists; omitted fixed fields; and signed
power-of-two-or-zero encoding.

### Key invariant

Differential tests compare intj's key with the final annotated `ASTSource`:

> same intj map key implies the same signature, constexpr mapping, and attrs

Tests cover dynamic-device module caches, fixed-device per-handle caches, and
the no-map single-kernel path. During implementation, mutating away each new
discriminator or check must make a targeted test fail.

### Runtime, binding, and compilation

AMD runtime tests cover forced scalar and pointer types, conditional allowlist
facts, `NEVER`, multiple facts, baked ordinary and constexpr values, bound
tensors and pointers, mutable tensor storage, null pointer behavior, device
binding, constexpr widths, and positive/negative power-of-two encodings.

Tests verify that each `bind()`/`bind_device()` call creates independent
`BoundLauncher` state, bound parameters disappear from signature metadata,
actual bound values do not change module identity, and fixed-device handles
do not share kernel records.

CUDA is compile-checked only because the development machine has no NVIDIA GPU.

### Generated check structure

Generated-source tests require one combined `INTJ_UNLIKELY` failure branch per
checked constrained argument and `INTJ_ASSUME` only after that branch. Unchecked
rendering must omit semantic verification code. Structural CPython safety and
driver-error checks remain in both modes.

### Performance benchmark

`benchmarks/bench_launch.py` supports:

```bash
python benchmarks/bench_launch.py
python benchmarks/bench_launch.py --no-gpu
```

The targeted matrix measures the current `AUTO` map path, reduced keys,
verification off/on, baked arguments, bound tensor/pointer calls, a
fixed-device map, and the fixed-device no-map path.

Every launcher is warmed before timing. The benchmark reports the median of
repeated batches, absolute time per call, delta from the matching baseline, and
construction/binding time separately. Results are informational; pytest checks
that host-only mode exercises the intended branches but applies no timing
threshold.

## Compatibility and rollout

- A kernel with no new annotations and default options preserves its existing
  dynamic argument and compilation semantics; `verify_annotation=False` emits
  no additional semantic checks.
- Bare `tl.constexpr` preserves its dynamic-kind 64-bit payload.
- `extra_annotation=None`, `bind_device=False`, and `no_gpu=False` remain the
  defaults.
- Unsupported or contradictory annotations fail loudly; there is no fallback
  to `JITFunction` launching.
- Major/minor package versioning plus template/runtime-header hashes prevent
  incompatible rendered extensions from being reused.
