# Compact Spec Key Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-encode the launcher's spec key from one 64-bit word per parameter to one byte per parameter, taking a representative kernel from 16 words to 5 and its cache slot from 144 bytes to 56.

**Architecture:** The key becomes a flat little-endian byte array: one code byte per declared parameter, one device byte, zero padding to a word boundary, then one 64-bit value word per `tl.constexpr` parameter. The pointer code carries a 5-bit *compact dtype index* — an index over the dtypes triton accepts, installed at load from a table `torch_abi.py` builds — rather than torch's raw `ScalarType` code, which is what lets the dtype, the divisibility bit and the pointer-range bit share one byte. `intj_hash` and the slot then specialize on `INTJ_NWORDS` behind unchanged signatures.

**Tech Stack:** C11/C++20 (`intj/runtime/intj_runtime.h`), Jinja2 (`intj/runtime/entry.c.jinja`), Python 3.12+, pytest, triton 3.8, torch 2.14, google/benchmark.

**Spec:** `docs/superpowers/specs/2026-09-23-compact-spec-key-design.md`

## Global Constraints

- **Test command:** `/tmp/gb2/bin/python -m pytest tests/ -q`. Baseline before any change: **77 passed, 1 skipped**. This env has torch 2.14.0+rocm7.2, triton 3.8.0, pytest, and a live AMD MI308X.
- **`intj_hash`, `intj_cache_get`, `intj_cache_put` keep the signature they have today in every `INTJ_NWORDS` fork** — `const uint64_t *` for the key, never a bare `uint64_t`. Both `tests/bench_kernel_cache.cpp` (lines 48, 59, 69, 87, 89, 117) and `entry.c.jinja` (lines 114, 236) call them; forking the signature forks both callers. All specialization lives below this API.
- **The key blob's length stays a multiple of 8.** `_validate_spec_key` unpacks it and `spec_key` returns it whole.
- **The blob must be byte-identical across `TorchAccess.SHIM`, `CPYTHON` and `CXX`** for the same arguments. `test_modes_agree_on_every_read` (test_launcher.py:647-660) enforces it.
- **`Param`, `RenderContext` and `ModuleKey` stay frozen, hashable and JSON-serializable.** No `bytes` and no enum fields — `ModuleKey.digest()` does `json.dumps(dataclasses.asdict(...))`.
- **`RenderContext`'s field set must exactly equal the template's variable set.** `launcher.py:385` renders under `jinja2.StrictUndefined`.
- Error contracts, unchanged: `OverflowError` for an int too large, `TypeError` for an unsupported argument type, `RuntimeError` for an unreadable tensor.
- Byte-code constants are defined once, in `intj_runtime.h`. Python never re-declares them — per `AGENTS.md`, `_validate_spec_key` compares raw bytes for equality and never interprets them.

## Review Focus

1. **A zero-element tensor of an unsupported dtype.** `intj_read_tensor`'s SHIM path returns at `intj_runtime.h:389` on `numel == 0` *before* its `code >= INTJ_NDTYPES` check at :395, and the CPYTHON reader has no bound check at all. Today that is harmless; under a byte code an unmapped dtype would alias. Must raise `RuntimeError` in all three modes. — Task 3.
2. **`device >= 256`.** One byte holds it, so device 256 aliases device 0 — a wrong-context launch. The check must sit with the device parse at `entry.c.jinja:61`, not with the key build: `entry.c.jinja:110` returns `None` for a zero-volume grid before the key exists. — Task 3.
3. **A kernel with exactly 7 versus 8 parameters.** `header_words = ceil((nparams + 1) / 8)`, so 7 params + device fills one word and 8 params + device needs two. Every kernel in the suite today has ≤ 8 parameters, so an off-by-one in the device byte passes everything. — Task 3.
4. **`torch.bool`, `torch.uint1` and `torch.int1` share compact index 8.** They all canonicalize to triton's `u1`, so they *must* key identically — merging is what keeps the key exactly as fine as triton's specialization rather than needlessly finer. — Task 1.
5. **A `dtype_index` table installed in only some access modes** reads as all-zero, keying every dtype to index 0 — silently coarse, and identical in all three modes, so the cross-mode test would not catch it. The third `set_torch_version` argument is mandatory in every mode. — Task 2.

---

## Task 1: The compact dtype index table

**Files:**
- Modify: `intj/torch_abi.py:1-5` (docstring), insert after `:107`
- Test: `tests/test_launcher.py` (new test beside `test_dtype_code_matches_torch` at :663)

**Interfaces:**
- Consumes: `NDTYPES` (`torch_abi.py:14`), `dtype_code()` (`torch_abi.py:64`)
- Produces: `torch_abi.dtype_index_table() -> bytes`, length `NDTYPES`, each entry `0..31` or `0xFF`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_launcher.py`, after `test_dtype_code_matches_torch`:

```python
def test_dtype_index_table_matches_triton():
    """The compact index must be exactly as fine as triton's specialization.

    Two torch dtypes share an index iff triton canonicalizes them to the same
    type -- torch.bool, torch.uint1 and torch.int1 are all `u1`.  Finer would
    cost redundant cache entries; coarser would launch the wrong kernel.
    """
    from triton._utils import type_canonicalisation_dict

    from intj.torch_abi import NDTYPES, dtype_code, dtype_index_table

    table = dtype_index_table()
    assert len(table) == NDTYPES

    canonical: dict[int, str] = {}
    for name in dir(torch):
        value = getattr(torch, name, None)
        if not isinstance(value, torch.dtype):
            continue
        canon = type_canonicalisation_dict.get(str(value).split(".")[-1])
        code = dtype_code(value)
        index = table[code]
        if canon is None:
            assert index == 0xFF, f"{value} is not a triton type but got index {index}"
            continue
        assert index < 32, f"{value} got index {index}, which does not fit 5 bits"
        assert canonical.setdefault(index, canon) == canon, (
            f"index {index} maps to both {canonical[index]} and {canon}"
        )

    assert canonical, "no torch dtype canonicalized; the table is empty"
    # the three names triton folds into one type must share one index
    assert table[dtype_code(torch.bool)] == table[dtype_code(torch.uint1)]
    assert table[dtype_code(torch.int1)] == table[dtype_code(torch.bool)]
    # and a dtype triton has never accepted must be refused
    assert table[dtype_code(torch.complex64)] == 0xFF


def test_dtype_index_table_is_deterministic():
    """Same torch, same triton, same table -- it is a ModuleKey input by proxy."""
    from intj.torch_abi import dtype_index_table

    dtype_index_table.cache_clear()
    first = dtype_index_table()
    dtype_index_table.cache_clear()
    assert dtype_index_table() == first
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/tmp/gb2/bin/python -m pytest tests/test_launcher.py -k dtype_index -q`
Expected: FAIL with `ImportError: cannot import name 'dtype_index_table'`

- [ ] **Step 3: Write the implementation**

In `intj/torch_abi.py`, insert after `itemsize_table()` ends (line 107), before `_probes()`:

```python
@functools.lru_cache(maxsize=1)
def dtype_index_table() -> bytes:
    """torch dtype code -> a 5-bit index over the dtypes triton accepts.

    The key encodes this rather than torch's raw `ScalarType` code so that the
    dtype, the divisibility bit and the pointer-range bit fit one byte: at six
    bits the pointer codes fill all 256 and leave nowhere for the scalar tags.
    Torch is already at 45 of its 64 codes and adds them faster than triton
    grows types, so the compact space also has the headroom the raw one does not.

    Two torch dtypes share an index exactly when triton canonicalizes them to
    the same type -- `bool`, `uint1` and `int1` are all `u1` -- which keeps the
    key as fine as triton's specialization and no finer.  `0xFF` means "triton
    does not take this dtype", and is the bound check the C side raises on.

    Built from the live torch and the installed triton, never baked in: what a
    module keys on has to describe the pair that is running now.
    """
    import torch
    from triton._utils import type_canonicalisation_dict

    by_code: dict[int, torch.dtype] = {}
    for name in dir(torch):
        value = getattr(torch, name, None)
        if not isinstance(value, torch.dtype):
            continue
        code = dtype_code(value)
        if 0 <= code < NDTYPES:
            by_code[code] = value

    table = bytearray(b"\xff" * NDTYPES)
    indices: dict[str, int] = {}
    for code in sorted(by_code):  # code order, so the assignment is stable
        # `str(dtype)` is what triton's own canonicalize_dtype splits; the
        # attribute name is not -- `torch.chalf` and `torch.complex32` are one
        # dtype, and only the latter spelling is ever a dict key.
        canon = type_canonicalisation_dict.get(str(by_code[code]).split(".")[-1])
        if canon is None:
            continue
        table[code] = indices.setdefault(canon, len(indices))

    if len(indices) > 32:  # pragma: no cover - triton would have to double
        raise RuntimeError(
            f"intj: triton now has {len(indices)} element types, which no longer "
            "fit the spec key's 5-bit dtype index"
        )
    return bytes(table)
```

Amend the module docstring at `intj/torch_abi.py:3-4`:

```python
"""How the generated module reaches torch.

Three strategies, and the discovery of what the shim one needs.  Everything here
is pure python and testable without a GPU.  Only `dtype_index_table` consults
triton, and lazily, the same way the rest consults torch.
"""
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/tmp/gb2/bin/python -m pytest tests/test_launcher.py -k dtype_index -q`
Expected: PASS, 2 tests

- [ ] **Step 5: Run the whole suite**

Run: `/tmp/gb2/bin/python -m pytest tests/ -q`
Expected: 79 passed, 1 skipped

- [ ] **Step 6: Commit**

```bash
git add intj/torch_abi.py tests/test_launcher.py
git commit -m "Add the compact dtype index the packed key will encode"
```

---

## Task 2: Install the table into every access mode

**Files:**
- Modify: `intj/runtime/intj_runtime.h:318-330` (`intj_torch_abi`)
- Modify: `intj/runtime/entry.c.jinja:266-343` (`set_torch_version`)
- Modify: `intj/launcher.py:27` (import), `:342` (call)
- Test: `tests/test_launcher.py:743-753`

**Interfaces:**
- Consumes: `torch_abi.dtype_index_table()` from Task 1
- Produces: `st->abi.dtype_index[INTJ_NDTYPES]`, readable by `INTJ_DECODE` in Task 3; `module.set_torch_version(version, layout, dtype_index)` — three positional arguments in all three modes

- [ ] **Step 1: Write the failing test**

Replace `test_unconfigured_module_refuses_to_launch` in `tests/test_launcher.py:743-753` — read the existing body first and keep its two negative assertions — and add beside it:

```python
def test_set_torch_version_needs_the_dtype_index_in_every_mode(axpy_launcher):
    """The dtype table is not a shim detail: all three readers key on it.

    Installed in only some modes, the others see an all-zero table and key every
    dtype to index 0 -- coarse in the same way in each, so the cross-mode
    equality test would not catch it.
    """
    from intj.torch_abi import NDTYPES, dtype_index_table

    module = axpy_launcher.__self__
    with pytest.raises(TypeError, match="set_torch_version"):
        module.set_torch_version((2, 14), None)  # two args is no longer enough
    with pytest.raises(ValueError, match="dtype index table"):
        module.set_torch_version((2, 14), None, b"\xff" * (NDTYPES - 1))
    assert len(dtype_index_table()) == NDTYPES
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/tmp/gb2/bin/python -m pytest tests/test_launcher.py -k set_torch_version -q`
Expected: FAIL — two arguments are still accepted, so `pytest.raises(TypeError)` does not fire

- [ ] **Step 3: Add the table to the ABI struct**

In `intj/runtime/intj_runtime.h`, inside `intj_torch_abi` (currently lines 318-330), after `itemsize`:

```c
  uint8_t itemsize[INTJ_NDTYPES];    /* dtype code -> element size */
  /* dtype code -> the spec key's 5-bit index, 0xFF where triton takes no such
   * dtype.  Unlike itemsize, every access mode reads this: it is what the key
   * encodes, not how a pointer is computed. */
  uint8_t dtype_index[INTJ_NDTYPES];
  int ready;                         /* set_torch_version has run */
```

- [ ] **Step 4: Take the third argument in all three modes**

In `intj/runtime/entry.c.jinja`, change `set_torch_version` (line 266 onward). Replace the arity check and `layout` fetch at lines 269-276 with:

```c
  intj_state *st = (intj_state *)PyModule_GetState(self);
  PyObject *layout;
  if (nargs != 3) {
    PyErr_SetString(PyExc_TypeError,
                    "intj: set_torch_version takes (version, layout, dtype_index)");
    return NULL;
  }
  layout = args[1];

  /* Every mode needs this, so it is parsed before the per-mode branch: a module
   * that skipped it would read an all-zero table and key every dtype alike. */
  {
    Py_ssize_t n = PyBytes_Size(args[2]);
    if (n < 0)
      return NULL;
    if (n != INTJ_NDTYPES) {
      PyErr_Format(PyExc_ValueError,
                   "intj: dtype index table must be %d bytes, got %zd",
                   INTJ_NDTYPES, n);
      return NULL;
    }
    memcpy(st->abi.dtype_index, PyBytes_AS_STRING(args[2]), INTJ_NDTYPES);
  }
```

Leave the three `{% if torch_access %}` branches and `st->abi.ready = 1;` untouched.

- [ ] **Step 5: Pass it from python**

In `intj/launcher.py:27`, add `dtype_index_table` to the `.torch_abi` import. Then at line 342:

```python
            module.set_torch_version(
                torch_version(), layout.as_args() if layout else None, dtype_index_table()
            )
```

- [ ] **Step 6: Run the tests**

Run: `/tmp/gb2/bin/python -m pytest tests/ -q`
Expected: 80 passed, 1 skipped. The table is installed but nothing reads it yet, so every existing key is unchanged.

- [ ] **Step 7: Commit**

```bash
git add intj/runtime/intj_runtime.h intj/runtime/entry.c.jinja intj/launcher.py tests/test_launcher.py
git commit -m "Install the dtype index in every access mode"
```

---

## Task 3: The packed key layout

The atomic one: the byte codes, the decode macros, the template's key build, `Param`, `nwords`, and `_validate_spec_key` all describe one encoding and cannot land apart.

**Files:**
- Modify: `intj/runtime/intj_runtime.h:618-755` (tags, `INTJ_WORD`, both decode macros)
- Modify: `intj/runtime/entry.c.jinja:8`, `:61-64`, `:113-127`, `:236-250`
- Modify: `intj/launcher.py:45-54` (`Param`), `:56-85` (`RenderContext`), `:165`, `:603-626`, `:712-740`
- Test: `tests/test_launcher.py`

**Interfaces:**
- Consumes: `st->abi.dtype_index` from Task 2
- Produces: `Param(name, is_constexpr, spec, align, cx_word)` where `cx_word: int | None`; `RenderContext.header_words: int`; `_render_params(jit_func) -> tuple[tuple[Param, ...], int]` returning params and `nwords`; `INTJ_DECODE(st, o, acc, shift, vals, np, SPEC, ALIGN, SBIT, pname)` and `INTJ_DECODE_CONSTEXPR(o, acc, shift, valword, pname)`

- [ ] **Step 1: Write the failing layout test**

Add to `tests/test_launcher.py`, near `_render_context` (~line 401):

```python
@pytest.mark.parametrize("nparams,nconstexpr", [(1, 0), (1, 1), (6, 0), (7, 0), (8, 0), (7, 2), (8, 3)])
def test_render_params_byte_layout(nparams, nconstexpr):
    """Byte i is parameter i; the device byte is last; constexpr words follow.

    The 7-vs-8 cases straddle the header word boundary -- 7 params plus the
    device byte fill one word exactly, 8 need a second.  Every kernel in this
    suite has at most 8 parameters, so nothing else here catches a device-byte
    off-by-one.
    """
    from intj.launcher import _render_params

    names = [f"p{i}" for i in range(nparams)]
    src = ", ".join(names + [f"C{i}: tl.constexpr" for i in range(nconstexpr)])
    namespace: dict[str, Any] = {"tl": tl, "triton": triton}
    exec(f"@triton.jit\ndef k({src}):\n    pass\n", namespace)

    params, nwords = _render_params(namespace["k"])
    header_words = -(-(nparams + nconstexpr + 1) // 8)

    assert [p.name for p in params] == names + [f"C{i}" for i in range(nconstexpr)]
    assert [p.cx_word for p in params if not p.is_constexpr] == [None] * nparams
    assert [p.cx_word for p in params if p.is_constexpr] == [
        header_words + i for i in range(nconstexpr)
    ]
    assert nwords == header_words + nconstexpr
```

- [ ] **Step 2: Run it to verify it fails**

Run: `/tmp/gb2/bin/python -m pytest tests/test_launcher.py -k render_params_byte -q`
Expected: FAIL with `AttributeError: 'Param' object has no attribute 'cx_word'` (and `_render_params` returning one value, not two)

- [ ] **Step 3: Replace the tag constants with byte codes**

In `intj/runtime/intj_runtime.h`, replace lines 618-637 (`INTJ_T_*`, `INTJ_FLAG_*`, `INTJ_WORD`) with:

```c
/* One byte per declared parameter.  The pointer range partitions the space, so
 * "is a pointer" needs no flag of its own, and `S` -- the one bit that is only
 * ever set for a pointer -- fits inside the byte rather than a section beside
 * it.  This is what the 5-bit compact dtype index buys: at torch's raw 6-bit
 * code the pointer codes alone would be all 256. */
#define INTJ_B_PTR(idx, d, s)                                                  \
  (((uint32_t)(idx) << 2) | ((uint32_t)(d) << 1) | (uint32_t)(s))
#define INTJ_B_I32 128u  /* +1 when tt.divisibility = 16 */
#define INTJ_B_I64 130u  /* +1 likewise */
#define INTJ_B_U64 132u  /* +1 likewise */
#define INTJ_B_FP32 134u
#define INTJ_B_U1 135u
#define INTJ_B_NONE 136u
#define INTJ_B_ONE 137u  /* int 1 folded to ("constexpr", 1) */
#define INTJ_B_CX_NONE 138u
#define INTJ_B_CX_BOOL 139u
#define INTJ_B_CX_INT 140u
#define INTJ_B_CX_UINT 141u
#define INTJ_B_CX_FLOAT 142u
/* 143..255 unused.  Constexpr codes are disjoint from the rest even though a
 * byte position is always one or the other at render time: it costs nothing and
 * turns a wrong offset into a nonsense code rather than a silent alias. */
```

- [ ] **Step 4: Rewrite `INTJ_DECODE`**

Replace `intj_runtime.h:655-720`. The tensor branch gains the index lookup — **the only dtype bound check that runs on every path, including `numel == 0`**:

```c
#define INTJ_DECODE(st, o, acc, shift, vals, np, SPEC, ALIGN, SBIT, pname)     \
  do {                                                                         \
    PyObject *_o = (o);                                                        \
    PyTypeObject *_t = Py_TYPE(_o);                                            \
    uint32_t _code;                                                            \
    if (_t == (st)->tensor_type || _t == (st)->param_type) {                   \
      void *_p = NULL;                                                         \
      int32_t _dt = -1;                                                        \
      int64_t _sz = 0;                                                         \
      int _want = (SPEC) && (SBIT);                                            \
      if (INTJ_UNLIKELY(intj_read_tensor(&(st)->abi, _o, &_p, &_dt, _want,     \
                                         &_sz) != 0)) {                        \
        intj_note_param(pname);                                                \
        return NULL;                                                           \
      }                                                                        \
      /* The readers disagree about how much they check: the shim one skips its \
       * own bound test for a zero-element tensor, and the cpython one has none. \
       * One lookup here covers all three modes and every path through them. */ \
      uint32_t _idx = (_dt >= 0 && _dt < INTJ_NDTYPES)                         \
                          ? (st)->abi.dtype_index[_dt]                         \
                          : 0xFFu;                                             \
      if (INTJ_UNLIKELY(_idx == 0xFFu)) {                                      \
        PyErr_Format(PyExc_RuntimeError,                                       \
                     "intj: tensor argument '%s' has dtype code %d, which "    \
                     "triton does not take",                                   \
                     pname, (int)_dt);                                         \
        return NULL;                                                           \
      }                                                                        \
      uint32_t _d = ((SPEC) && (ALIGN) && (((uintptr_t)_p & 15u) == 0)) ? 1u : 0u; \
      uint32_t _s = (_want && _sz <= 2147483647LL) ? 1u : 0u;                  \
      _code = INTJ_B_PTR(_idx, _d, _s);                                        \
      (vals)[(np)] = (uint64_t)(uintptr_t)_p;                                  \
      (np)++;                                                                  \
    } else if (_o == Py_True || _o == Py_False) {                              \
      _code = INTJ_B_U1;                                                       \
      (vals)[(np)] = (uint64_t)(_o == Py_True);                                \
      (np)++;                                                                  \
    } else if (PyLong_CheckExact(_o)) {                                        \
      uint64_t _bits;                                                          \
      int _kind = intj_as_int(_o, &_bits);                                     \
      if (INTJ_UNLIKELY(_kind == INTJ_INT_TOO_BIG)) {                          \
        PyErr_Format(PyExc_OverflowError,                                      \
                     "intj: integer argument '%s' is too large", pname);       \
        return NULL;                                                           \
      }                                                                        \
      int64_t _v = (int64_t)_bits;                                             \
      if (_kind == INTJ_INT_I64 && (SPEC) && _v == 1) {                        \
        _code = INTJ_B_ONE;                                                    \
      } else {                                                                 \
        uint32_t _d = ((SPEC) && (ALIGN) && ((_bits & 15u) == 0)) ? 1u : 0u;   \
        _code = (_kind == INTJ_INT_U64          ? INTJ_B_U64                   \
                 : (_v >= INT32_MIN && _v <= INT32_MAX) ? INTJ_B_I32           \
                                                        : INTJ_B_I64) +        \
                _d;                                                            \
        (vals)[(np)] = _bits;                                                  \
        (np)++;                                                                \
      }                                                                        \
    } else if (PyFloat_CheckExact(_o)) {                                       \
      float _f = (float)((PyFloatObject *)_o)->ob_fval;                        \
      uint32_t _bits;                                                          \
      memcpy(&_bits, &_f, 4);                                                  \
      _code = INTJ_B_FP32;                                                     \
      (vals)[(np)] = (uint64_t)_bits;                                          \
      (np)++;                                                                  \
    } else if (_o == Py_None) {                                                \
      _code = INTJ_B_NONE;                                                     \
    } else {                                                                   \
      PyErr_Format(PyExc_TypeError,                                            \
                   "intj: unsupported argument '%s' of type %s; pass a "       \
                   "torch.Tensor, int, float, bool or None",                   \
                   pname, Py_TYPE(_o)->tp_name);                               \
      return NULL;                                                             \
    }                                                                          \
    (acc) |= _code << (shift);                                                 \
  } while (0)
```

- [ ] **Step 5: Rewrite `INTJ_DECODE_CONSTEXPR`**

Replace `intj_runtime.h:722-755`:

```c
/* Decode a tl.constexpr argument into one code byte and one value word. */
#define INTJ_DECODE_CONSTEXPR(o, acc, shift, valword, pname)                   \
  do {                                                                         \
    PyObject *_o = (o);                                                        \
    uint32_t _code;                                                            \
    if (_o == Py_None) {                                                       \
      _code = INTJ_B_CX_NONE;                                                  \
      (valword) = 0;                                                           \
    } else if (_o == Py_True || _o == Py_False) {                              \
      _code = INTJ_B_CX_BOOL;                                                  \
      (valword) = (uint64_t)(_o == Py_True);                                   \
    } else if (PyLong_CheckExact(_o)) {                                        \
      uint64_t _bits;                                                          \
      int _kind = intj_as_int(_o, &_bits);                                     \
      if (INTJ_UNLIKELY(_kind == INTJ_INT_TOO_BIG)) {                          \
        PyErr_Format(PyExc_OverflowError,                                      \
                     "intj: constexpr argument '%s' is too large", pname);     \
        return NULL;                                                           \
      }                                                                        \
      _code = _kind == INTJ_INT_U64 ? INTJ_B_CX_UINT : INTJ_B_CX_INT;          \
      (valword) = _bits;                                                       \
    } else if (PyFloat_CheckExact(_o)) {                                       \
      double _d = ((PyFloatObject *)_o)->ob_fval;                              \
      _code = INTJ_B_CX_FLOAT;                                                 \
      memcpy(&(valword), &_d, 8);                                              \
    } else {                                                                   \
      PyErr_Format(PyExc_TypeError,                                            \
                   "intj: unsupported constexpr argument '%s' of type %s; "    \
                   "pass an int, float, bool or None",                         \
                   pname, Py_TYPE(_o)->tp_name);                               \
      return NULL;                                                             \
    }                                                                          \
    (acc) |= _code << (shift);                                                 \
  } while (0)
```

- [ ] **Step 6: Rewrite the key build in the template**

In `intj/runtime/entry.c.jinja`, replace the `entry` key block at lines 113-127:

```c
  /* spec key + kernel arguments.  The header accumulates in 32-bit registers --
   * the width the decode already works in -- and each key word gets exactly one
   * store: two adjacent 32-bit stores feeding an 8-byte load would not forward. */
  uint64_t key[INTJ_NWORDS];
  uint64_t vals[INTJ_NSLOTS];
  int np = 0;
{% for a in range(header_words * 2) %}  uint32_t _a{{ a }} = 0;
{% endfor %}
{% for p in params %}
{% if p.is_constexpr %}
  INTJ_DECODE_CONSTEXPR(args[{{ 3 + loop.index0 }}], _a{{ loop.index0 // 4 }},
                        {{ (loop.index0 % 4) * 8 }}, key[{{ p.cx_word }}],
                        intj_param_names[{{ loop.index0 }}]);
{% else %}
  INTJ_DECODE(st, args[{{ 3 + loop.index0 }}], _a{{ loop.index0 // 4 }},
              {{ (loop.index0 % 4) * 8 }}, vals, np,
              {{ p.spec }}, {{ p.align }}, {{ spec_pointer_range }},
              intj_param_names[{{ loop.index0 }}]);
{% endif %}
{% endfor %}
  _a{{ (params | length) // 4 }} |= (uint32_t)device << {{ ((params | length) % 4) * 8 }};
{% for w in range(header_words) %}  key[{{ w }}] = (uint64_t)_a{{ 2 * w }} | ((uint64_t)_a{{ 2 * w + 1 }} << 32);
{% endfor %}
```

Apply the same block to `spec_key` at lines 236-250, with `args[{{ loop.index0 }}]` instead of `args[{{ 3 + loop.index0 }}]` and `uint64_t device = 0;` in place of the parsed device.

- [ ] **Step 7: Bound the device**

In `intj/runtime/entry.c.jinja`, at lines 61-64 — with the parse, not the key build, because line 110 returns for a zero-volume grid before any key exists:

```c
  int64_t device;
  if (INTJ_UNLIKELY(!PyLong_CheckExact(args[0]) ||
                    intj_as_i64(args[0], &device) != 0 || device < 0 ||
                    device > 255)) {
    PyErr_SetString(PyExc_TypeError,
                    "intj: device must be an int in [0, 256)");
    return NULL;
  }
```

- [ ] **Step 8: Update `Param` and `_render_params`**

In `intj/launcher.py`, replace the `word` field (line 53):

```python
    cx_word: int | None  # value-word index; None unless is_constexpr
```

Replace `_render_params` (lines 603-626) — it now returns `nwords` too, so the formula lives in one place instead of being mirrored at line 165:

```python
def _render_params(jit_func: JitFunction) -> tuple[tuple[Param, ...], int]:
    """One render-time descriptor per declared kernel parameter, and the key length.

    Byte `i` of the key is parameter `i`, the device byte follows them, and the
    constexpr value words follow the padding.  `nwords` comes back with the
    params because it is the same arithmetic: computing it apart from the offsets
    is how the two drift.
    """
    declared = list(jit_func.params)
    header_words = -(-(len(declared) + 1) // 8)
    params: list[Param] = []
    cx = header_words
    for p in declared:
        kind = p._param.kind
        if kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            raise UnsupportedKernel(f"intj: parameter {p.name!r} is {kind}; only positional parameters are supported")
        if not p.is_constexpr and p.annotation:
            raise UnsupportedKernel(
                f"intj: parameter {p.name!r} has annotation {p.annotation!r}; only tl.constexpr annotations "
                "are supported"
            )
        params.append(
            Param(
                name=p.name,
                is_constexpr=p.is_constexpr,
                spec=0 if p.do_not_specialize else 1,
                align=0 if p.do_not_specialize_on_alignment else 1,
                cx_word=cx if p.is_constexpr else None,
            )
        )
        if p.is_constexpr:
            cx += 1
    return tuple(params), cx
```

At the call site, `intj/launcher.py:148`, unpack both values:

```python
    params, nwords = _render_params(jit_func)
```

Add `header_words: int` to `RenderContext` beside `nwords` (line 67), and at lines 165-166:

```python
        nwords=nwords,
        header_words=-(-(len(params) + 1) // 8),
        max_slots=sum(0 if p.is_constexpr else 1 for p in params),
```

- [ ] **Step 9: Update `_validate_spec_key`**

In `intj/launcher.py`, replace lines 728-735. `import struct` at line 11 becomes unused — delete it:

```python
    specialization = triton_specialization(jit_func, args, options)

    for i, param in enumerate(params):
        mine: tuple[int, ...] = (keyblob[i],)
        if param.cx_word is not None:
            offset = param.cx_word * 8
            mine += (int.from_bytes(keyblob[offset:offset + 8], "little"),)
        theirs = repr(specialization[i])
```

- [ ] **Step 10: Fix the fixture, then run the layout test**

In `tests/test_launcher.py:392-401`, switch `_render_context`'s `Param(...)` to keywords so a future field change fails loudly instead of mis-assigning, and correct the sizes — one pointer plus one constexpr is `header_words = 1`, `nwords = 2`:

```python
        params=(
            Param(name="x", is_constexpr=False, spec=1, align=1, cx_word=None),
            Param(name="BLOCK", is_constexpr=True, spec=1, align=1, cx_word=1),
        ),
        nwords=2, header_words=1, max_slots=1, spec_pointer_range=1,
```

Run: `/tmp/gb2/bin/python -m pytest tests/test_launcher.py -k render_params_byte -q`
Expected: PASS, 7 cases

- [ ] **Step 11: Write the two Review Focus tests**

Place both beside `test_modes_agree_on_every_read` (test_launcher.py:647-660), whose per-mode construction the first one reuses. `scale` is `scale(x, o, n, s, BLOCK)` (test_launcher.py:49) and `axpy` is `axpy(x, y, o, n, a, flag, bias, BLOCK)` (:37).

```python
def test_unsupported_dtype_is_refused_in_every_mode():
    """The path that skipped the dtype check: numel == 0 returns before it.

    intj_read_tensor's shim reader bails at numel == 0 ahead of its own bound
    test, and the cpython reader has no bound test at all.  Harmless while the
    dtype had 32 bits of the key to itself; under a byte code every unmapped
    dtype would alias every other one.  So the check lives in the decode, which
    all three readers pass through.
    """
    modules = {
        m: getattr(make_launcher(scale, torch_access=m), "__self__") for m in ACCESS_MODES
    }
    o = torch.empty(4096, device="cuda")
    for mode, module in modules.items():
        for x in (torch.zeros(4, device="cuda", dtype=torch.complex64),
                  torch.zeros(0, device="cuda", dtype=torch.complex64)):
            with pytest.raises(RuntimeError, match="triton does not take"):
                module.spec_key(x, o, 1024, 2.0, 128)


def test_device_above_255_is_refused(axpy_launcher):
    """One byte holds the device, so 256 would alias 0 -- a wrong-context launch."""
    x = torch.randn(64, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    for device in (256, -1):
        with pytest.raises(TypeError, match=r"\[0, 256\)"):
            axpy_launcher(device, stream, (1,), x, x, x, 64, 1.5, True, None, 128)
```

- [ ] **Step 12: Run the whole suite**

Run: `/tmp/gb2/bin/python -m pytest tests/ -q`
Expected: all pass. `test_spec_key_is_never_coarser_than_triton` is the one that matters — it asserts the new encoding never maps two triton specializations to one key.

- [ ] **Step 13: Confirm the sizes actually shrank**

Run:

```bash
/tmp/gb2/bin/python -c "
import torch, triton, triton.language as tl
from intj.launcher import _render_params
@triton.jit
def k(a, b, c, d, e, f, n0, n1, n2, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr):
    pass
print('nwords:', _render_params(k)[1], '(spec says 5)')
@triton.jit
def add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    pass
print('add_kernel nwords:', _render_params(add_kernel)[1], '(spec says 2)')
"
```

Expected: `nwords: 5` and `add_kernel nwords: 2`. If either differs, the device byte or `header_words` is off — stop and fix before committing.

- [ ] **Step 14: Commit**

```bash
git add intj/runtime/intj_runtime.h intj/runtime/entry.c.jinja intj/launcher.py tests/test_launcher.py
git commit -m "Pack the spec key to one byte per parameter"
```

---

## Task 4: Measure the new key sizes

Parametrizing the benchmark *before* the hash and slot work means Tasks 5 and 6 each show a delta rather than a claim.

**Files:**
- Modify: `tests/test_kernel_cache.py:27`, `:46-62`, `:76-95`, `:98-100`
- Modify: `tests/bench_kernel_cache.cpp:1-16` (doc comment only)

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces: benchmark coverage at `INTJ_NWORDS` 1, 2 and 5

- [ ] **Step 1: Parametrize over nwords**

Read `tests/test_kernel_cache.py` first. Replace line 27:

```python
#: 1 = no constexpr and <= 7 params; 2 = add_kernel(x, y, out, n, BLOCK_SIZE);
#: 5 = six tensors, three ints, three constexpr.  All three hash and slot
#: specializations have to compile and self-check, not just the one this box
#: happens to run.
_NWORDS = (1, 2, 5)
```

Give `_build` an `nwords` parameter and use it at line 51 in place of the module constant.

- [ ] **Step 2: Widen the results key**

In the build/run loop (lines 76-87), nest per-nwords, name the binary `tmp / f"bench_{cache.value}_{nwords}"`, and key results on `(nwords, entry["name"])`. Two binaries emit identical benchmark names, so without this the second `nwords` silently overwrites the first rather than failing.

Print one table per `nwords` with the header `f"kernel cache, {nwords}-word key, ns per operation"`, and update the unpack in the `0 < ns < 1000` assertion at lines 98-100.

- [ ] **Step 3: Note what each size covers**

In `tests/bench_kernel_cache.cpp`'s doc comment (lines 1-16), say which sizes are measured and why — 1 is the keyless 16-byte slot with a bijective hash, 2 is the single multiply, 5 is the loop. Correct the build line at line 7 to show the three.

No C++ source change: `intj_hash` and `intj_cache_get/put` keep their `const uint64_t *` signatures in every fork, so lines 35, 48, 59, 69, 87, 89 and 117 compile unchanged at every size.

- [ ] **Step 4: Run and record**

Run: `/tmp/gb2/bin/python -m pytest tests/test_kernel_cache.py -q -s`
Expected: three tables, or a skip if google/benchmark is not installed. Paste the numbers into the commit message — they are the baseline Tasks 5 and 6 improve on.

- [ ] **Step 5: Commit**

```bash
git add tests/test_kernel_cache.py tests/bench_kernel_cache.cpp
git commit -m "Benchmark the key sizes the packed layout actually produces"
```

---

## Task 5: Specialize the hash on the key size

**Files:**
- Modify: `intj/runtime/intj_runtime.h:102-108`

**Interfaces:**
- Consumes: `intj_mix` (`intj_runtime.h:97`)
- Produces: `intj_hash(const uint64_t *w)` — **same signature in every fork**

- [ ] **Step 1: Replace `intj_hash`**

```c
/* The renderer knows INTJ_NWORDS, so the shape is chosen rather than looped.
 *
 * At one word the mix is replaced by a bijection -- xorshift-right and an odd
 * multiply both are -- which makes hash equality key equality and lets the slot
 * drop the key altogether.  At two, one multiply suffices, the way wyhash
 * handles a short input.  Above that the chain is split in two lanes: a mix is
 * a ~4-cycle multiply and the whole chain sits between the last decode and the
 * first probe, so at five words that is ~20 cycles of pure latency serially and
 * ~12 in pairs.  Same reason wyhash runs three lanes over a 48-byte block.
 */
#if INTJ_NWORDS == 1
static inline uint64_t intj_hash(const uint64_t *w) {
  uint64_t x = w[0];
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebull;
  x ^= x >> 31;
  return x;
}
#elif INTJ_NWORDS == 2
static inline uint64_t intj_hash(const uint64_t *w) {
  const uint64_t s0 = 0xa0761d6478bd642full, s1 = 0xe7037ed1a0b428dbull;
  return intj_mix(intj_mix(w[0] ^ s0, w[1] ^ s1), 2 * 8 + s1);
}
#else
static inline uint64_t intj_hash(const uint64_t *w) {
  const uint64_t s0 = 0xa0761d6478bd642full, s1 = 0xe7037ed1a0b428dbull;
  uint64_t h0 = s0, h1 = s1;
  int i = 0;
  for (; i + 1 < INTJ_NWORDS; i += 2) {
    h0 = intj_mix(h0 ^ s1, w[i] ^ s0);
    h1 = intj_mix(h1 ^ s0, w[i + 1] ^ s1);
  }
  if (i < INTJ_NWORDS)
    h0 = intj_mix(h0 ^ s1, w[i] ^ s0);
  return intj_mix(h0 ^ h1, INTJ_NWORDS * 8 + s1);
}
#endif
```

- [ ] **Step 2: Run the suite and the benchmark**

Run: `/tmp/gb2/bin/python -m pytest tests/ -q -s`
Expected: all pass. `hash_only` should drop at every size; compare against Task 4's numbers.

- [ ] **Step 3: Commit**

```bash
git add intj/runtime/intj_runtime.h
git commit -m "Choose the hash shape from the key size"
```

---

## Task 6: Drop the stored key at one word

**Files:**
- Modify: `intj/runtime/intj_runtime.h:117-135` (`intj_slot`, `intj_key_eq`), `:137-168` (`intj_map_get`, `intj_map_insert`)

**Interfaces:**
- Consumes: the bijective `intj_hash` from Task 5
- Produces: `intj_map_get`, `intj_map_insert`, `intj_map_put` — same signatures

- [ ] **Step 1: Fork the slot**

Replace `intj_slot` (lines 117-121):

```c
/* Task 5's one-word hash is a bijection, so `hash == h` IS key equality and the
 * slot has no reason to carry the key: 16 bytes, four to a cache line, against
 * 8 + 8 + INTJ_NWORDS * 8. */
#if INTJ_NWORDS == 1
typedef struct {
  uint64_t hash;
  intj_kernel *val; /* NULL => empty slot */
} intj_slot;
#else
typedef struct {
  uint64_t hash;
  intj_kernel *val; /* NULL => empty slot */
  uint64_t key[INTJ_NWORDS];
} intj_slot;
#endif
```

- [ ] **Step 2: Fork the two accessors**

In `intj_map_get` (line 144), drop the second conjunct at one word:

```c
#if INTJ_NWORDS == 1
    if (INTJ_LIKELY(s->hash == h))
#else
    if (INTJ_LIKELY(s->hash == h && intj_key_eq(s->key, k)))
#endif
      return s->val;
```

In `intj_map_insert` (line 166), guard the `memcpy` the same way. Keep the `const uint64_t *k` parameter in both — the signature is a Global Constraint, and at one word `k` is simply unread. Add `(void)k;` inside the `#if` so `-Wunused-parameter` stays quiet.

`intj_key_eq` (lines 133-135) is now unreferenced at one word. Guard the whole function with `#if INTJ_NWORDS > 1` rather than leaving a dead `static inline`.

Leave the tsl and absl backends alone: `struct intj_key { uint64_t w[1]; }` already *is* a bare `uint64_t` in layout, and `memcmp` of a literal 8 is one compare. Keying them on a scalar would be a second code path for no measured gain.

- [ ] **Step 3: Run the suite and the benchmark**

Run: `/tmp/gb2/bin/python -m pytest tests/ -q -s`
Expected: all pass. The `1`-word `hit` and `miss` rows should drop against Task 5's numbers at the larger `Arg`s, where the slot count per cache line is what moves.

Note when reading them: `hit/1` and `miss/1` index with `& 0`, so they always touch element 0 and measure a permanently hot line. Read `hit/64` and `hit/512`.

- [ ] **Step 4: Commit**

```bash
git add intj/runtime/intj_runtime.h
git commit -m "Drop the stored key where the hash is a bijection"
```

---

## Task 7: Reconcile the docs with what was built

**Files:**
- Modify: `docs/superpowers/specs/2026-09-23-compact-spec-key-design.md`
- Modify: `intj/kernel_cache.py:5-12` (measured table)
- Modify: `intj/launcher.py:124-138` (`make_launcher` docstring)

- [ ] **Step 1: Correct the spec's two wrong claims**

The Testing section says `bench_kernel_cache.cpp` is "currently built at 5 and 11". It is built at 5 only — `test_kernel_cache.py:27` was a scalar, and the 11 comes from `intj_key_eq`'s comment at `intj_runtime.h:129`. Change it to record 5 → (1, 2, 5).

The Design section says the table assigns "indices in ScalarType code order to those present in triton's `type_canonicalisation_dict`", which gives 19. What was built indexes triton's *canonical type*, giving 17 — `bool`, `uint1` and `int1` share `u1`. Record that, and why: it makes the key exactly as fine as triton's specialization instead of needlessly finer.

- [ ] **Step 2: Document the device bound**

`make_launcher`'s docstring (lines 124-138) describes `launcher(device, stream, grid, ...)`. Add that `device` must be in `[0, 256)`.

- [ ] **Step 3: Refresh the measured table**

`intj/kernel_cache.py:5-12` heads its numbers `ns/lookup, 40-byte key`. Replace with the figures Task 6 produced, at the sizes the packed layout actually produces.

- [ ] **Step 4: Run the suite one last time**

Run: `/tmp/gb2/bin/python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add docs/ intj/kernel_cache.py intj/launcher.py
git commit -m "Record what the packed key measured"
```
