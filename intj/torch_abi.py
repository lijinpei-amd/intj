"""How the generated module reaches torch.

Three strategies, and the discovery of what the shim one needs.  Everything here
is pure python and testable without a GPU; nothing here imports triton.
"""

import ctypes
import dataclasses
import enum
import functools
from typing import Any

#: How many dtype codes the C side reserves room for; mirrors INTJ_NDTYPES.
NDTYPES = 64

#: Bytes of TensorImpl / StorageImpl the probe is willing to read.  Both are far
#: larger than this in every torch that has shipped -- TensorImpl's
#: `sizes_and_strides_` alone is 88 bytes and sits in the middle -- but the
#: windows are kept as small as the search allows, because reading past a live
#: object is the one mistake here that cannot be caught.
_TENSORIMPL_WINDOW = 176
_STORAGEIMPL_WINDOW = 56


class TorchAccess(enum.Enum):
    """How a generated module reads a tensor's pointer, dtype and storage size."""

    #: Pick by what the running torch and toolchain support.
    AUTO = "auto"
    #: Call through the interpreter.  Assumes nothing but THPDtype, which self-checks.
    CPYTHON = "cpython"
    #: Read torch's structs at offsets discovered at load.  No torch code runs.
    SHIM = "shim"
    #: Compile the extension as C++ against torch's headers.
    CXX = "cxx"


@dataclasses.dataclass(frozen=True)
class TensorLayout:
    """Byte offsets the `SHIM` mode reads, plus the dtype-to-element-size table.

    Discovered from the running process rather than from a table keyed on
    `torch.__version__`: the offsets are then true by construction for whatever
    torch is loaded, including one intj has never seen.
    """

    cdata: int  # PyObject*    -> TensorImpl*
    storage: int  # TensorImpl*  -> StorageImpl*
    storage_offset: int  # TensorImpl*  -> int64
    numel: int  # TensorImpl*  -> int64
    data_type: int  # TensorImpl*  -> TypeMeta index, low byte
    s_data: int  # StorageImpl* -> void*
    s_nbytes: int  # StorageImpl* -> int64
    itemsize: bytes  # dtype code -> element size, NDTYPES long

    def as_args(self) -> tuple[Any, ...]:
        """The tuple `set_torch_version` parses."""
        return (
            self.cdata, self.storage, self.storage_offset, self.numel,
            self.data_type, self.s_data, self.s_nbytes, self.itemsize,
        )


def dtype_code(dtype: Any) -> int:
    """The `at::ScalarType` value of a `torch.dtype`, read off the singleton.

    `struct THPDtype { PyObject_HEAD at::ScalarType scalar_type; char name[65]; }`
    has been byte-identical since torch 1.13, and `ScalarType` is an
    `enum class : int8_t`.  The C side makes the same read and validates it
    against the embedded `name[]`; this is its python twin.
    """
    return ctypes.c_int8.from_address(id(dtype) + _pyobject_size()).value


@functools.lru_cache(maxsize=1)
def _pyobject_size() -> int:
    """`sizeof(PyObject)`: refcount plus type pointer, whatever the build."""
    return ctypes.sizeof(ctypes.c_void_p) * 2


@functools.lru_cache(maxsize=1)
def itemsize_table() -> bytes:
    """dtype code -> element size, for every dtype this torch can allocate.

    Built from the live torch rather than hardcoded, so a torch that adds a dtype
    needs no change here.  A zero entry means "no such dtype", which the C side
    treats as a layout mismatch rather than computing a pointer from a guess.
    """
    import warnings

    import torch

    table = bytearray(NDTYPES)
    for name in dir(torch):
        value = getattr(torch, name, None)
        if not isinstance(value, torch.dtype):
            continue
        code = dtype_code(value)
        if not 0 <= code < NDTYPES:
            continue
        try:
            with warnings.catch_warnings():  # experimental/deprecated dtypes warn
                warnings.simplefilter("ignore")
                table[code] = torch.empty(0, dtype=value).element_size()
        except Exception:  # pragma: no cover - dtype the build cannot allocate
            continue
    return bytes(table)


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


def _window(address: int, size: int) -> bytes:
    return ctypes.string_at(address, size)


def _find_u64(blob: bytes, want: int, align: int = 8) -> set[int]:
    """Offsets in `blob` holding `want` as a little-endian 8-byte value."""
    target = want.to_bytes(8, "little", signed=want < 0)
    return {i for i in range(0, len(blob) - 7, align) if blob[i:i + 8] == target}


def _find_u8(blob: bytes, want: int) -> set[int]:
    return {i for i in range(len(blob)) if blob[i] == want}


def probe_layout() -> TensorLayout | None:
    """Discover the `SHIM` offsets from the running torch, or return None.

    Each offset is the intersection, over every probe tensor, of the positions
    whose value matches what torch's own python accessors report.  Anything that
    does not come down to exactly one candidate yields None -- the caller then
    refuses the shim mode rather than reading a guessed offset, which is the one
    failure here that cannot raise.
    """
    import torch

    head = _pyobject_size()
    cdata: set[int] | None = None
    ti_fields: dict[str, set[int]] = {}
    si_fields: dict[str, set[int]] = {}

    for t in _probes():
        storage = t.untyped_storage()
        impl, simpl = t._cdata, storage._cdata
        if not impl or not simpl:
            return None

        obj = _window(id(t), head + 8 * 4)
        ti = _window(impl, _TENSORIMPL_WINDOW)
        si = _window(simpl, _STORAGEIMPL_WINDOW)

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

    layout = TensorLayout(itemsize=itemsize_table(), **pinned)
    return layout if _selfcheck(layout) else None


def _selfcheck(layout: TensorLayout) -> bool:
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


def _read(layout: TensorLayout, t: Any) -> tuple[int, int, int]:
    """What the C shim reader would compute, in python."""
    impl = ctypes.c_size_t.from_address(id(t) + layout.cdata).value
    simpl = ctypes.c_size_t.from_address(impl + layout.storage).value
    numel = ctypes.c_int64.from_address(impl + layout.numel).value
    code = ctypes.c_uint8.from_address(impl + layout.data_type).value
    nbytes = ctypes.c_int64.from_address(simpl + layout.s_nbytes).value if simpl else 0
    if numel == 0 or not simpl:
        return (0, code, nbytes)
    data = ctypes.c_size_t.from_address(simpl + layout.s_data).value
    off = ctypes.c_int64.from_address(impl + layout.storage_offset).value
    return (data + off * layout.itemsize[code], code, nbytes)


def _expected(t: Any) -> tuple[int, int, int]:
    return (t.data_ptr(), dtype_code(t.dtype), t.untyped_storage().nbytes())


@functools.lru_cache(maxsize=1)
def cached_layout() -> TensorLayout | None:
    """`probe_layout()` once per process; torch cannot change under us."""
    return probe_layout()


@functools.lru_cache(maxsize=1)
def torch_version() -> tuple[int, int]:
    import torch

    major, _, rest = torch.__version__.partition(".")
    minor = rest.partition(".")[0]
    return (int(major), int(minor))
