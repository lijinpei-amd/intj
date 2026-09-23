# Argument annotations

## Problem

`intj.make_launcher()` currently accepts only Triton's existing unannotated
arguments and bare `tl.constexpr` parameters. Its `extra_annotation` argument is
reserved and rejected.

That leaves three useful facts unavailable to the generated launcher:

- The set, or exact value, of runtime types a parameter may have.
- Whether Triton's normal value, alignment, and pointer-range specialization is
  automatic, disabled, or guaranteed by the caller.
- The storage width of a constexpr value, including the compact case where an
  integer is known to be zero or a power of two.

The annotation must affect both compilation and module identity. A rendered
module may omit an assumed fact from its per-launch map key only because it
checks that fact on every launch and because a module built with a different
annotation has a different `ModuleKey.digest()`.

The safety invariant remains:

> same intj spec key implies the same Triton specialization

Here, "Triton specialization" means the annotated `ASTSource` signature,
constexpr values, and specialization attributes that intj actually compiles.

## Goals

- Add an intj-owned annotation hierarchy for normal and constexpr arguments.
- Accept annotations inline and through `extra_annotation`, with deterministic
  field-by-field merging.
- Keep annotations immutable, canonical, hashable, and JSON-serializable once
  they enter `RenderContext`.
- Let annotations change the signature, constexprs, and specialization facts
  presented to Triton's compiler without mutating the original `JITFunction`.
- Reduce constexpr key storage when a declared type or
  `power_of_two_or_zero` permits it.
- Keep satisfied launch-time constraints on a small, predictable C fast path.

## Non-goals

- Supporting arbitrary Python type annotations.
- Casting or reinterpreting tensor storage.
- Adding scalar conversion code for fp8, fp16, or bf16.
- Encoding assumed facts in the rendered module's per-launch map key.
- Replacing Triton's type inference for unannotated or allowlisted arguments.
- Supporting new Triton backends, keyword arguments, variadic parameters, or
  other currently refused kernel shapes.

## Public API

The public definitions live in `intj/annotation.py` and are re-exported from
`intj`:

```python
from dataclasses import dataclass
from typing import Sequence

import triton.language as tl


@dataclass(frozen=True)
class Annotation: ...


@dataclass(frozen=True)
class Specialization: ...


@dataclass(frozen=True)
class Fact: ...


@dataclass(frozen=True)
class Argument(Annotation):
    type: tl.dtype | Sequence[tl.dtype] | None = None
    specialize: Specialization | None = None


@dataclass(frozen=True)
class Constexpr(Annotation):
    type: tl.dtype | Sequence[tl.dtype] | None = None
    power_of_two_or_zero: bool = False
```

`Sequence` inputs are normalized to tuples during construction. Public
annotation objects therefore remain immutable and hashable even when the caller
passes a list.

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

`Assume` accepts one or more facts. Version one supports the specialization
facts Triton 3.8 exposes to this launcher:

- `EqualTo(1)`
- `Aligned(16)`
- `PointerRange(32)`

Other values are rejected rather than accepted with no compiler effect.

Frequently used type sets are immutable tuples:

```python
INT_TYPES = (tl.int32, tl.int64, tl.uint64)
FLOAT_TYPES = (tl.float32,)
```

There is no `UINT_TYPES` preset. `tl.uint1` is boolean-like and remains an
explicit type rather than a member of `INT_TYPES`.

Two existing Triton annotations are shorthands:

```python
tl.int32      # Argument(type=tl.int32)
tl.constexpr  # Constexpr()
```

Examples:

```python
@triton.jit
def kernel(
    x: Argument(type=tl.pointer_type(tl.float32)),
    n: Argument(type=INT_TYPES, specialize=Assume(Aligned(16))),
    BLOCK: Constexpr(type=tl.int32, power_of_two_or_zero=True),
):
    ...


launcher = intj.make_launcher(
    kernel,
    extra_annotation={
        "n": Argument(type=INT_TYPES),
    },
)
```

The equal `INT_TYPES` field is accepted while the inline specialization field
is retained.

## Accepted annotation sources

Each declared parameter may receive fields from:

1. Its inline Python annotation.
2. `extra_annotation[name]` passed to `make_launcher()`.
3. Triton's existing `do_not_specialize` and
   `do_not_specialize_on_alignment` decorator state.

Inline and extra values accept only:

- An intj `Annotation` instance.
- A Triton `tl.dtype`, as shorthand for `Argument(type=...)`.
- Bare `tl.constexpr`, as shorthand for `Constexpr()`.

`extra_annotation` is a mapping from declared parameter name to one of those
values. Unknown names are rejected.

Fields merge independently. `None` means unspecified and may be filled by the
other source. Equal duplicate fields are accepted. Unequal duplicate fields,
incompatible facts, and disagreement with Triton's decorator flags are
rejected.

After merging, every unspecified specialization fact becomes `AUTO`.
Explicit `AUTO` is a specified value: it conflicts with a decorator flag that
requires that fact to be disabled. This distinction lets `None` remain the
field-merging sentinel.

The merged result, not either input object, is canonicalized and hashed.

## Canonical representation and module identity

Each rendered `Param` stores one immutable canonical annotation containing:

- The kind: `Argument` or `Constexpr`.
- `None` or a tuple of canonical Triton type names.
- The mode of every supported specialization fact: `AUTO`, `NEVER`, or
  `ASSUME`.
- The `power_of_two_or_zero` flag.
- The computed key payload width and byte offset.

Type and fact collections are deduplicated and sorted by stable canonical
names. Ordering in a caller-provided list therefore has no semantic or hashing
effect.

`RenderContext.params` contains these canonical values. Because `ModuleKey`
already serializes and hashes the complete `RenderContext`, no second annotation
field is added to `ModuleKey`.

The existing `ModuleKey.params` decorator tuple must not retain a raw,
unmerged annotation representation. It either omits that item or uses the same
canonical value from `Param`; otherwise equivalent annotations could hash
differently or fail JSON serialization.

For a fixed JIT function and source, equivalent annotation inputs share a
rendered module. Any canonically different merged annotation produces a
different render and digest, even when the generated machine code would happen
to be identical.

## Argument type semantics

### Inference and allowlists

`Argument(type=None)` preserves Triton's unspecialized runtime type inference.

A single type forces that signature type. The launch path verifies the Python
value kind and exact representable range before packing it.

A type sequence is an allowlist, not a list of coercion candidates. Intj asks
Triton for the unspecialized inferred type and requires that canonical type to
be a member of the list.

### Supported scalar types

Scalar `Argument.type` accepts:

- All Triton integer dtypes.
- `tl.uint1` for Python `bool`.
- `tl.float32` and `tl.float64` for Python `float`.

Python `bool` is accepted only for `tl.uint1`; it is not treated as an integer.
Integer and floating-point values are not implicitly converted into each other.

Low-precision scalar floating types (`fp8`, `fp16`, and `bf16`) are rejected.
They require rounding and bit-packing machinery that is outside this feature.

### Pointer types

A tensor type is written with native Triton meaning:

```python
tl.pointer_type(tl.float32)
```

Bare `tl.float32` always means a scalar. A runtime tensor must match the
canonical pointer element type exactly. Intj does not cast or reinterpret tensor
storage. Any pointer element type jointly supported by the live Torch and
Triton pair is accepted because only the pointer value is passed through.

A type list may contain both scalar and pointer types. Runtime inference selects
one kind and exact canonical membership decides whether the launch is accepted.

A typed `Argument` rejects `None`. Only `Argument(type=None)` retains intj's
current inferred `None` behavior.

## Constexpr type semantics

Bare `Constexpr()` and bare `tl.constexpr` preserve the current dynamic-kind,
64-bit-payload behavior. They accept the same `int`, `float`, `bool`, and `None`
values as today.

A typed `Constexpr` accepts integer dtypes, `tl.uint1`, and `tl.float64`.
It rejects fp8, fp16, bf16, and fp32. A Python float is binary64; storing only
its rounded fp32 representation while passing the original binary64 value to
Triton could map two different compiler constants to one intj key.

A single constexpr type requires exact representability. A type list accepts a
value representable by at least one member and reserves a payload as wide as
the largest member. The parameter's descriptor byte records the selected
canonical kind when signedness or member identity is needed to avoid aliases.
When several members represent the value, selection is deterministic: smallest
width first, then canonical type-name order.

The original Python constexpr value is passed to Triton unchanged. The declared
type controls validation and key storage only; it does not introduce lossy
constexpr conversion.

Typed constexpr rejects `None`. `bool` is accepted only by `tl.uint1`.

### Power of two or zero

`power_of_two_or_zero` is valid only on `Constexpr`. The runtime value must be an
exact Python integer, not a `bool`, and must also satisfy any declared constexpr
type constraint.

The encoding is:

```text
0      -> 0
2**n   -> n + 1
```

Negative values, non-powers of two, and values outside Triton's accepted
constexpr range are rejected before lookup. The encoded exponent uses the
parameter's 8-bit descriptor/value field, so this form allocates no separate
constexpr payload.

## Specialization semantics

Specialization is modeled independently for equality-to-one, 16-byte
alignment, and 32-bit pointer range after all sources are merged.

`AUTO` uses Triton's normal decision for each launch.

`NEVER` disables value, alignment, and pointer-range specialization. Triton's
existing decorator flags are folded into the corresponding internal fact modes.
`do_not_specialize` disables equality-to-one and alignment while leaving
pointer range automatic, matching Triton 3.8. `do_not_specialize_on_alignment`
affects alignment only. Public `NEVER` is deliberately stronger and disables
all three facts in the annotated `ASTSource`.

`Assume(...)` has four effects:

1. Every named fact is checked on every launch.
2. Triton compiles with the fact present.
3. The fact is omitted from the rendered module's per-launch map key.
4. Facts not named by `Assume` remain available for another merged source and
   otherwise become `AUTO`.

`Assume(EqualTo(1))` compiles the value as a constant and removes that parameter
from the GPU argument slots. `Aligned(16)` applies to integer or pointer values.
`PointerRange(32)` applies only to pointers and means the backing storage fits
the backend's signed 32-bit pointer-range specialization (at most 2 GiB minus
one byte, matching Triton 3.8). When `type=None`, the fact itself narrows the
accepted runtime values and an inapplicable value fails at launch.

Multiple assumed facts are allowed. Statically impossible combinations, such
as `EqualTo(1)` with `Aligned(16)` on the same value, are rejected during
launcher creation.

## Per-launch key layout

The key is a fixed-size byte array whose final size is rounded up to 64 bits for
the existing hash and cache interface. Fields are placed by descending
alignment to avoid internal padding:

```text
[64-bit constexpr payloads]
[32-bit constexpr payloads]
[16-bit constexpr payloads]
[8-bit fields: parameter descriptors/values and 8-bit constexpr payloads]
[device byte]
[zero padding to 64 bits]
```

Within each width group, parameter-derived fields remain in declaration order.
If one parameter contributes both a descriptor and an 8-bit payload, the
descriptor comes first. The device byte follows all parameter fields.

Every parameter has an 8-bit descriptor. It records the runtime type or
automatic specialization facts that may vary within one rendered module.
Ordinary constexpr parameters additionally receive a payload sized as follows:

- Untyped constexpr: 64 bits.
- Typed constexpr: the largest width in its canonical type set.
- `power_of_two_or_zero`: no payload; the descriptor stores the compact value.

All payloads use little-endian bit representations. Signed integers use their
declared-width two's-complement representation. A descriptor distinguishes
otherwise ambiguous signed and unsigned list members.

`AUTO` facts participate in the map key because they can change Triton's
specialization between launches. `NEVER` facts contribute no varying key state.
`ASSUME` facts also contribute no key state: their values are fixed by the
rendered module's canonical annotation and enforced by launch-time checks.

The annotation itself is never copied into the per-launch map key. It belongs
to `RenderContext` and therefore to `ModuleKey.digest()`.

## Launch-time validation

Generated C validates runtime kinds, forced-type ranges, allowlist membership,
typed constexpr ranges, and assumed facts before cache lookup or compilation.

All facts for one constrained argument share one fast-path branch:

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
compiler builtin where available and is emitted only after the checked failure
branch, so it cannot remove validation.

Checks reuse values already decoded for the key and GPU argument array. The
satisfied path performs no Python callback, allocation, cache lookup, or error
formatting beyond the normal launch path.

## Compilation flow

The original `JITFunction` is never mutated, and no replacement `JITFunction`
is constructed. Constructing a duplicate registers it globally under the same
module and qualified name, which would disturb Triton's registry and other
users of the original function.

On a cache miss:

1. The C extension passes the original Python arguments to the existing Python
   compile callback.
2. Intj invokes Triton's existing binder to obtain the baseline specialization.
   This is required only for `AUTO` behavior and type-list inference.
3. Intj applies the canonical annotation to produce the final signature,
   constexpr mapping, and specialization attributes. Forced types replace the
   inferred type; `Constexpr` moves the original value into the constants;
   `NEVER` removes facts; and `Assume` retains or adds the facts already checked
   by generated C.
4. Intj constructs `ASTSource(jit_func, signature, constexprs, attrs)` and
   calls Triton's compiler through the existing compilation machinery.
5. Intj initializes and retains the resulting `CompiledKernel` exactly as it
   does today.

The intj map key is not an input to compilation. The callback receives it only
for the existing cold-path invariant validator. That validator compares the key
against the final annotated `ASTSource` specialization, not the original binder
result.

## Errors

No new exception hierarchy is added.

- `TypeError` during annotation construction: a field has the wrong Python
  kind.
- `ValueError` during annotation construction: an empty type list, unsupported
  dtype, unsupported fact value, or internally contradictory facts.
- `ValueError` during `make_launcher`: an unknown parameter or a conflict among
  inline, extra, and Triton decorator sources.
- `UnsupportedKernel`: the resulting valid annotation is used on a kernel or
  parameter shape intj does not support.
- `TypeError` at launch: the runtime Python value has the wrong kind.
- `ValueError` or `OverflowError` at launch: allowlist rejection, failed fact,
  invalid power-of-two-or-zero value, or numeric range failure.

Every launch-time annotation error occurs before cache lookup and names the
parameter and failed constraint.

## Verification

### Canonicalization and value types

Tests cover inline/extra merging, equal duplicates, reordered and duplicate
types/facts, contradictions, immutable equality/hash behavior, and JSON
serialization. Equivalent merged annotations produce equal contexts and module
digests; canonically different annotations do not.

### Layout

Layout tests cover every boundary between 64-, 32-, 16-, and 8-bit groups;
declaration order within groups; the device byte; final padding; largest-member
reservation for type lists; and the payload-free power-of-two-or-zero encoding.

### Key invariant

The differential test compares intj's key with the final annotated
`ASTSource` specialization:

> same intj map key implies the same signature, constexpr mapping, and attrs

Invalid assumed facts must fail before `spec_key` returns a key. During
implementation, mutating away each new discriminator or check must make a
targeted test fail; a test that cannot detect the corresponding coarsening is
not sufficient.

### Runtime and compilation

AMD runtime tests cover forced scalar and pointer types, allowlist acceptance
and rejection, `NEVER`, each assumed fact, multiple facts, typed constexpr
boundaries, zero and powers of two, error timing, and GPU argument counts.

CUDA is compile-checked only because the development machine has no NVIDIA GPU.

### Fast path

Generated-source tests require one combined `INTJ_UNLIKELY` failure branch per
constrained argument, `INTJ_ASSUME` only after that branch, and no Python call,
allocation, or map-key bytes for assumed facts.

`benchmarks/bench_launch.py` gains cached-launch and `spec_key` cases with zero,
one, and multiple satisfied facts. It reports the median delta but does not set
a timing threshold in CI; a nanosecond-scale threshold would be less stable
than the structural checks above.

## Compatibility and rollout

- A kernel with no new annotations preserves current argument and compilation
  semantics.
- Bare `tl.constexpr` preserves its dynamic-kind 64-bit payload.
- `extra_annotation=None` remains the default.
- Unsupported or contradictory annotations fail loudly; there is no fallback
  to `JITFunction` launching.
- The key-layout and render-context changes require a `SCHEMA_VERSION` bump so
  no existing rendered extension is reused under the new layout.
