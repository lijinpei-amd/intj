# PyObject-relative tensor `cdata`

## Problem

`RUNTIME_SHIM` reads the TensorImpl pointer through a recorded offset in a
PyTorch tensor's Python object. The 13 rows in `torch_abi.toml` currently record
an **absolute** offset measured on a GIL build with a 16-byte `PyObject` header.
`layout_for()` replaces those 16 bytes with `object.__basicsize__` before the
module loads. This mixes a CPython fact into the Torch ABI table and prevents
`abi_detect` from generating entries on a free-threaded build, whose header is
32 bytes on supported x86-64 CPython.

The interfaces already have separate owners: `python_intf.cpython_abi.pyobject_size()`
reports the detecting interpreter's header size, while each generated module
is compiled against the running interpreter's `Python.h`. Its compiled
`sizeof(PyObject)` is therefore the header size that must be used for the
module's pointer read. `python_intf.check` compares the reported and compiled
sizes on each supported build.

## Decision and data contract

Keep the field name `cdata` and the eight-element `TensorABI.as_args()` tuple,
but change the field's meaning at the table and Python boundary. No new mode or
per-launch callback is needed.

| Location | Meaning of `cdata` |
| --- | --- |
| `torch_abi.toml`, `TensorABI`, `probe_layout()`, `cpp_detect.detect()`, setter argument | Nonnegative bytes from the end of `PyObject_HEAD` to the TensorImpl pointer slot. |
| `intj_torch_abi.cdata` in module state | Absolute bytes from the `PyObject *` to that slot. |

At module load, `set_torch_version()` computes and saves

```text
absolute_cdata = sizeof(PyObject) + recorded_cdata
```

The existing launch read, `object + st->abi.cdata`, stays unchanged. The
compiled module's size is authoritative for that read; Python's
`object.__basicsize__` is used to normalize detector output and to refuse an
unrepresentable offset before building a module. This is a one-time addition,
so the launch path does not gain work.

`RUNTIME_SHIM` continues to reuse its binary across verified PyTorch versions,
but **not** across CPython ABIs. `ModuleKey` includes the exact CPython version,
free-threaded build flag, extension suffix, and generated template bytes.
`STATIC_COMPILE` still obtains `THPVariable` offsets from C++ headers;
`INTERPRETER` still calls Python for tensor pointer and storage size. The
integer and float readers in `python_intf` do not change.

## Detection and table migration

The Python detector receives `header_size` from
`python_intf.cpython_abi.pyobject_size()`. It searches the live tensor object for an
**absolute** pointer slot, as it does today, and uses that absolute address
for its safety-bounded probe. Once a unique slot is pinned, it refuses a slot
before `header_size`, subtracts `header_size`, and self-checks by adding the
same size back for a live read. The command-line generator accepts both GIL
and free-threaded builds; its 16-byte generation guard is removed.

The independent C++ detector keeps `measure()`'s raw facts, including compiled
`sizeof(PyObject)` and the absolute final pointer-slot offset. `detect()`
checks that the final offset is at or after the header, then subtracts that
compiled size before comparing with the table. The final offset includes the
nested `MaybeOwned<Tensor>` field on PyTorch before 2.10; subtracting only from
`offsetof(THPVariable, cdata)` would be wrong.

The existing rows were all measured with a documented 16-byte header. Their
schema migration is exact arithmetic: the eight PyTorch 2.2–2.9 rows change
`24 → 8`, and the five 2.10–2.14 rows change `16 → 0`. Regenerate each row with
the updated detector, assert that arithmetic and every other recorded field
against the old row, then install the generated output. Dtype lists, version
keys, and strict parser rules stay as they are. The C++ detector independently
checks the new rows.

## Refusal and installation

`layout_for()` still requires a recorded version, an exact live dtype match,
and a 64-bit pointer. It returns the relative `TensorABI` only when
`pyobject_size() + recorded_cdata` fits in the C module's `uint16_t`; otherwise
explicit `RUNTIME_SHIM` selection raises `UnsupportedKernel` before building.
This Python check preserves the early refusal. The setter independently
checks `sizeof(PyObject) + relative_cdata <= UINT16_MAX` before writing any
module state, so even direct calls cannot truncate the offset. Negative or
non-integer offsets remain invalid, while relative zero is valid.

The launcher installs the tuple once under its existing load lock, before
publishing the module. It does not pass a CPython size argument: the compiled
setter already knows that size. A failed installation leaves the module's ABI
state unchanged. Changing `entry.c.jinja` changes `ModuleKey`'s digest, so old
cached binaries cannot be loaded with the new relative table.

## Verification and scope

- Compare the table's relative row with the live Python probe and the
  independent C++ detector; reconstruct an absolute address only for direct
  pointer reads in tests. Keep the parser's zero and 16-bit boundary checks.
- Test the C setter with a relative offset that fits in 16 bits but overflows
  after adding `sizeof(PyObject)`, and verify that a failed call changes no
  installed state. The CPU stub launch must pass the real tensor data pointer.
- Run the Python matrix on the supported GIL builds (3.8–3.14) and
  free-threaded builds (3.13t, 3.14t). Run the C++ detector on at least one
  GIL and one free-threaded build; the matrix itself does not invoke it.
- Run the full launcher suite, including all Torch access modes and
  `test_spec_key_is_never_coarser_than_triton`, on the available AMD GPU.

The table remains an x86-64 measurement. This work does not add another CPU
architecture or a CPython runtime shim. It also does not change the separate
non-x86-64 refusal work tracked in the root `TODO.md`.

## Alternatives considered

Keeping the 16-byte baseline would retain the generator's GIL-only
restriction. Recording a relative value but adding `object.__basicsize__` in
Python would remove that restriction, yet the module would still receive an
absolute offset computed outside the binary that performs the read. Using the
compiled `sizeof(PyObject)` at installation puts the CPython ABI fact at its
owner and requires no new launch-time operation.
