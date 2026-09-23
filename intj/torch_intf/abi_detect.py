"""Detect the torch ABI at runtime, from the torch that is running.

Every offset is found by matching field values against what torch's own python
accessors report, so an entry generated this way is verified by construction.
Needs no compiler; `cpp_detect` is the independent check that does.

    python -m intj.torch_intf.abi_detect   # prints a torch_abi.toml entry
"""

import ctypes
import functools
from typing import Any

from .torch_abi import (
    OFFSETS,
    TensorABI,
    dtype_code,
    itemsize_bytes,
    live_dtypes,
    pyobject_size,
    torch_version,
)

#: The most of TensorImpl / StorageImpl the probe reads.  Wide enough for every
#: field it looks for in every torch since 2.2 (`data_type_` sits at 176 in 2.2),
#: small enough that a wider search does not invite coincidences.  Each read is
#: further capped at the object's own allocation -- TensorImpl is 176 bytes in
#: torch 2.13, so this window alone would read past it, and reading past a live
#: object is the one mistake here that cannot be caught.
_TENSORIMPL_WINDOW = 192
_STORAGEIMPL_WINDOW = 72



def _probes() -> list[Any]:
    """Tensors whose disagreement pins every offset to one candidate.

    The set is load-bearing, not decoration:

    * different dtypes and numels keep `numel_`, `storage_offset_` and
      `data_type_` from being pinned by a coincidence,
    * a view with a non-zero `storage_offset` separates `storage_offset_` from
      the many zero fields around it,
    * `frombuffer` separates `StorageImpl::data_` from `DataPtr::ctx_`, which are
      equal for every allocator torch owns -- CPU and the HIP caching allocator
      both -- so a probe set of only torch-allocated tensors leaves the two tied
      forever and picking wrong is silently correct until someone passes a
      borrowed buffer.
    """
    import torch

    base = torch.arange(4096, dtype=torch.float16)
    wide = torch.arange(300, dtype=torch.float64)
    borrowed = torch.frombuffer(bytearray(4096), dtype=torch.uint8)
    return [
        base,
        base[3:],
        base[1000:2000],
        wide,
        wide[7:],
        torch.arange(64, dtype=torch.int32)[5:],
        torch.tensor(1.0),
        borrowed,
        borrowed[17:],
    ]


@functools.lru_cache(maxsize=1)
def _malloc_usable_size() -> Any:
    fn = ctypes.CDLL(None).malloc_usable_size
    fn.argtypes, fn.restype = [ctypes.c_void_p], ctypes.c_size_t
    return fn


def _heap_window(address: int, size: int) -> bytes:
    """Up to `size` bytes of the `new`-allocated object at `address`, never past it."""
    return _window(address, min(size, _malloc_usable_size()(address)))


def _window(address: int, size: int) -> bytes:
    return ctypes.string_at(address, size)


def _find_u64(blob: bytes, want: int, align: int = 8) -> set[int]:
    """Offsets in `blob` holding `want` as a little-endian 8-byte value."""
    target = want.to_bytes(8, "little", signed=want < 0)
    return {i for i in range(0, len(blob) - 7, align) if blob[i:i + 8] == target}


def _find_u8(blob: bytes, want: int) -> set[int]:
    return {i for i in range(len(blob)) if blob[i] == want}


def probe_layout() -> TensorABI | None:
    """Discover the offsets from the running torch, or return None.

    Not used when launching: `_LAYOUTS` is consulted instead, so a torch intj has
    not been verified against is refused rather than read at guessed offsets.
    This is what *generates* those entries (`python -m intj.torch_intf.abi_detect`)
    and what the test suite checks them against.

    Each offset is the intersection, over every probe tensor, of the positions
    whose value matches what torch's own python accessors report.  Anything that
    does not come down to exactly one candidate yields None -- the caller then
    refuses the shim mode rather than reading a guessed offset, which is the one
    failure here that cannot raise.
    """
    head = pyobject_size()
    cdata: set[int] | None = None
    ti_fields: dict[str, set[int]] = {}
    si_fields: dict[str, set[int]] = {}

    for t in _probes():
        storage = t.untyped_storage()
        impl, simpl = t._cdata, storage._cdata
        if not impl or not simpl:
            return None

        obj = _window(id(t), head + 8 * 4)
        ti = _heap_window(impl, _TENSORIMPL_WINDOW)
        si = _heap_window(simpl, _STORAGEIMPL_WINDOW)

        found = {
            "storage": _find_u64(ti, simpl),
            "storage_offset": _find_u64(ti, t.storage_offset()),
            "numel": _find_u64(ti, t.numel()),
            "data_type": _find_u8(ti, dtype_code(t.dtype)),
        }
        s_found = {
            "s_data": _find_u64(si, storage.data_ptr()),
            "s_nbytes": _find_u64(si, storage.nbytes()),
        }
        here = _find_u64(obj, impl)
        cdata = here if cdata is None else (cdata & here)
        for key, hits in found.items():
            ti_fields[key] = hits if key not in ti_fields else (ti_fields[key] & hits)
        for key, hits in s_found.items():
            si_fields[key] = hits if key not in si_fields else (si_fields[key] & hits)

    pinned: dict[str, int] = {}
    for name, hits in [("cdata", cdata or set()), *ti_fields.items(), *si_fields.items()]:
        if len(hits) != 1:
            return None
        pinned[name] = hits.pop()

    # the live dtypes, not the recorded ones: this is what *generates* an entry,
    # so it has to run on a torch that has none yet
    layout = TensorABI(itemsize=itemsize_bytes(live_dtypes()), **pinned)
    return layout if _selfcheck(layout) else None


def _selfcheck(layout: TensorABI) -> bool:
    """Reproduce torch's own `data_ptr()` through the discovered offsets.

    The probe pins each offset independently; this checks the arithmetic that
    combines them, including the zero-element rule that `data_ptr()` applies and
    the raw offsets cannot see.
    """
    import torch

    cases = list(_probes())
    cases += [
        torch.zeros(0, dtype=torch.float16),
        torch.zeros(4096, dtype=torch.float16)[4096:],  # empty, live storage
        torch.arange(8, dtype=torch.bfloat16)[3:],
    ]
    if torch.cuda.is_initialized():  # probing must not be what starts the GPU
        gpu = torch.arange(1024, device="cuda", dtype=torch.float32)
        cases += [gpu, gpu[7:], gpu[1024:]]
    return all(_read(layout, t) == _expected(t) for t in cases)


def _read(layout: TensorABI, t: Any) -> tuple[int, int, int]:
    """What the C shim reader would compute, in python."""
    impl = ctypes.c_size_t.from_address(id(t) + layout.cdata).value
    simpl = ctypes.c_size_t.from_address(impl + layout.storage).value
    numel = ctypes.c_int64.from_address(impl + layout.numel).value
    code = ctypes.c_uint8.from_address(impl + layout.data_type).value
    if not simpl:
        raise RuntimeError("tensor has no storage")
    nbytes = ctypes.c_int64.from_address(simpl + layout.s_nbytes).value
    if numel == 0:
        return (0, code, nbytes)
    data = ctypes.c_size_t.from_address(simpl + layout.s_data).value
    off = ctypes.c_int64.from_address(impl + layout.storage_offset).value
    return (data + off * layout.itemsize[code], code, nbytes)


def _expected(t: Any) -> tuple[int, int, int]:
    return (t.data_ptr(), dtype_code(t.dtype), t.untyped_storage().nbytes())


def _main() -> None:
    """Print the entry for the running torch, to paste into `torch_abi.toml`."""
    import torch

    layout = probe_layout()
    if layout is None:
        raise SystemExit(f"intj: cannot pin torch {torch.__version__}'s layout")

    dtypes = live_dtypes()
    # `at::ScalarType` is a dense enum, so a gap means a code this torch uses is
    # named nowhere in its namespace -- and a tensor of that dtype would be
    # refused at launch as a layout mismatch.  Say so here rather than print an
    # entry with a hole in it.
    holes = sorted(set(range(max(dtypes, default=0))) - set(dtypes))
    if holes:
        raise SystemExit(f"intj: torch {torch.__version__} names no dtype for codes {holes}")

    print(f"# torch {torch.__version__}, x86-64")
    print('["%d.%d"]' % torch_version())
    for f in OFFSETS:
        print(f"{f} = {getattr(layout, f)}")
    print("dtypes = [  # indexed by at::ScalarType code")
    for code, (name, size) in sorted(dtypes.items()):
        print(f'  ["{name}", {size}],  # {code}')
    print("]")


if __name__ == "__main__":
    _main()
