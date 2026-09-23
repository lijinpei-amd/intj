"""How the generated module reaches torch.

Three strategies, and the discovery of what the shim one needs.  Everything here
is pure python and testable without a GPU.  Only `dtype_index_table` consults
triton, and lazily, the same way the rest consults torch.
"""

import ctypes
import dataclasses
import enum
import functools
import pathlib
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
    #: Read torch's structs at verified offsets.  No torch code runs.
    SHIM = "shim"
    #: Compile the extension as C++ against torch's headers.
    CXX = "cxx"


@dataclasses.dataclass(frozen=True)
class TensorABI:
    """Byte offsets the `SHIM` mode reads, plus the dtype-to-element-size table.

    Both halves come from one stanza of `torch_abi.txt`, keyed on the torch
    version: the offsets were measured on a torch with exactly those dtypes, so
    they are verified together or not at all.
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
def live_dtypes() -> dict[int, tuple[str, int]]:
    """dtype code -> (torch's own spelling, element size), from the running torch.

    `str(dtype)` rather than the attribute name: `torch.half` and `torch.float16`
    are one dtype with one canonical spelling, and only that spelling is stable
    enough to compare a recorded row against.

    `dtype.itemsize` reads the `TypeMeta` directly, so it answers for dtypes this
    build cannot actually allocate and, unlike a zero-element `torch.empty`,
    warns for none of the experimental ones.
    """
    import torch

    found: dict[int, tuple[str, int]] = {}
    for name in dir(torch):
        value = getattr(torch, name, None)
        if not isinstance(value, torch.dtype):
            continue
        code = dtype_code(value)
        if 0 <= code < NDTYPES:
            found[code] = (str(value).split(".")[-1], value.itemsize)
    return found


def _itemsize_bytes(dtypes: dict[int, tuple[str, int]]) -> bytes:
    """The `NDTYPES`-long table the C side indexes by dtype code.

    A zero entry means "no such dtype", which the C side treats as a layout
    mismatch rather than computing a pointer from a guess.
    """
    table = bytearray(NDTYPES)
    for code, (_, size) in dtypes.items():
        table[code] = size
    return bytes(table)


@functools.lru_cache(maxsize=4)
def itemsize_table(version: tuple[int, int] | None = None) -> bytes | None:
    """The recorded dtype -> element size table for `version`, or None.

    None both when intj has no row for this torch and when it has one the running
    torch disagrees with.  The disagreement matters as much as the absence: the
    offsets in the same stanza were measured against *that* set of dtypes, and a
    torch that has since renamed, dropped or inserted one is a torch those
    offsets were never verified on.  Refusing sends the caller to `CPYTHON`,
    which assumes no layout at all.
    """
    recorded = _DTYPES.get(version or torch_version())
    if recorded is None or recorded != live_dtypes():
        return None
    return _itemsize_bytes(recorded)


@functools.lru_cache(maxsize=1)
def dtype_index_table() -> bytes:
    """torch dtype code -> a 5-bit index over the dtypes triton accepts.

    The spec key encodes this rather than torch's raw `ScalarType` code so that
    the dtype, the divisibility bit and the pointer-range bit fit in one byte: at
    six bits the pointer codes alone fill all 256 and leave nowhere for the
    scalar tags.  Torch is already at 45 of its 64 codes and adds them faster
    than triton grows element types, so the compact space also has the headroom
    the raw one no longer does.

    Two torch dtypes share an index exactly when triton canonicalizes them to the
    same type -- `bool`, `uint1` and `int1` are all `u1` -- which keeps the key
    as fine as triton's specialization and no finer.  `0xFF` means "triton does
    not take this dtype", and is the bound check the C side raises on.

    Built from the live torch and the installed triton, never hardcoded: what a
    module keys on has to describe the pair that is running now.
    """
    import torch
    from triton._utils import type_canonicalisation_dict

    by_code: dict[int, Any] = {}
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
        # attribute name is not.  `torch.chalf` and `torch.complex32` are one
        # dtype, and only the latter spelling is ever a key in that dict.
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


def probe_layout() -> TensorABI | None:
    """Discover the offsets from the running torch, or return None.

    Not used when launching: `_LAYOUTS` is consulted instead, so a torch intj has
    not been verified against is refused rather than read at guessed offsets.
    This is what *generates* those rows (`python -m intj.torch_abi`) and what the
    test suite checks them against.

    Each offset is the intersection, over every probe tensor, of the positions
    whose value matches what torch's own python accessors report.  Anything that
    does not come down to exactly one candidate yields None -- the caller then
    refuses the shim mode rather than reading a guessed offset, which is the one
    failure here that cannot raise.
    """
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

    # the live dtypes, not the recorded ones: this is what *generates* a stanza,
    # so it has to run on a torch that has none yet
    layout = TensorABI(itemsize=_itemsize_bytes(live_dtypes()), **pinned)
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


#: The verified torch ABIs, as a data file beside this module.
#:
#: Text rather than a dict literal, and a separate file rather than a string in
#: here, so that a stanza is pasted exactly as `_main` printed it -- with no
#: chance of it being re-indented, re-ordered or "tidied" on the way in.
#:
#: Every number in it must be *measured*, never reasoned about: run
#: `python -m intj.torch_abi` on the target torch and paste what it prints.
#: `probe_layout()` finds each offset by matching field values against what
#: torch's own accessors report, so a stanza generated that way is verified by
#: construction.  `test_hardcoded_layout_matches_this_torch` re-checks the
#: stanza for whatever torch the suite runs against.
_ABI_PATH = pathlib.Path(__file__).with_name("torch_abi.txt")

#: Field names of `TensorABI` that the `layout:` row carries, in `_main`'s order.
_OFFSETS = tuple(f.name for f in dataclasses.fields(TensorABI) if f.name != "itemsize")


def _parse_abi(text: str) -> dict[tuple[int, int], dict[str, str]]:
    """Stanzas of `[major.minor]` followed by `name: value` rows.

    A line that starts with whitespace continues the row above it, so a row too
    long for one line can be wrapped anywhere without the parser caring where.
    `#` starts a comment; blank lines are ignored.
    """
    stanzas: dict[tuple[int, int], dict[str, str]] = {}
    rows: dict[str, str] | None = None
    key: str | None = None
    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        if line.startswith("["):
            if not line.endswith("]"):
                raise ValueError(f"intj: torch_abi.txt has an unclosed header {line!r}")
            major, minor = line[1:-1].split(".")
            rows, key = {}, None
            stanzas[(int(major), int(minor))] = rows
        elif rows is None:
            raise ValueError(f"intj: torch_abi.txt has {line!r} before any [version]")
        elif raw[:1].isspace():
            if key is None:
                raise ValueError(f"intj: torch_abi.txt continues nothing at {line!r}")
            rows[key] = f"{rows[key]} {line}"
        else:
            key, _, value = line.partition(":")
            rows[key] = value.strip()
    return stanzas


def _row(rows: dict[str, str], name: str, version: tuple[int, int]) -> str:
    if name not in rows:
        raise ValueError("intj: the torch %d.%d stanza has no %s row" % (*version, name))
    return rows[name]


def _parse_offsets(row: str, version: tuple[int, int]) -> dict[str, int]:
    """A `layout:` row: `name=offset ...`, every field of `_OFFSETS` once."""
    offsets = dict(pair.split("=") for pair in row.split())
    if tuple(offsets) != _OFFSETS:
        raise ValueError("intj: the torch %d.%d layout row has the wrong fields" % version)
    return {k: int(v) for k, v in offsets.items()}


def _parse_dtypes(row: str) -> dict[int, tuple[str, int]]:
    """A `dtypes:` row: `code=name:size ...`, in the shape `live_dtypes` returns."""
    dtypes: dict[int, tuple[str, int]] = {}
    for entry in row.split():
        code, _, spelling = entry.partition("=")
        name, _, size = spelling.partition(":")
        dtypes[int(code)] = (name, int(size))
    return dtypes


_ABI = _parse_abi(_ABI_PATH.read_text())
_LAYOUTS = {v: _parse_offsets(_row(r, "layout", v), v) for v, r in _ABI.items()}
_DTYPES = {v: _parse_dtypes(_row(r, "dtypes", v)) for v, r in _ABI.items()}


def supported_versions() -> list[tuple[int, int]]:
    return sorted(_LAYOUTS)


@functools.lru_cache(maxsize=4)
def layout_for(version: tuple[int, int] | None = None) -> TensorABI | None:
    """The layout for `version`, or None if intj has no verified stanza for it.

    None rather than a guess: a wrong offset cannot raise, it reads whatever
    happens to be at that address and hands the kernel a pointer built from it.
    A stanza whose `dtypes:` row no longer describes the running torch counts as
    no stanza at all -- see `itemsize_table`.
    """
    version = version or torch_version()
    offsets = _LAYOUTS.get(version)
    items = itemsize_table(version)
    if offsets is None or items is None:
        return None
    return TensorABI(itemsize=items, **offsets)


@functools.lru_cache(maxsize=1)
def torch_version() -> tuple[int, int]:
    import torch

    major, _, rest = torch.__version__.partition(".")
    minor = rest.partition(".")[0]
    return (int(major), int(minor))


def _main() -> None:
    """Print the stanza for the running torch, to paste into `torch_abi.txt`."""
    import textwrap

    import torch

    layout = probe_layout()
    if layout is None:
        raise SystemExit(f"intj: cannot pin torch {torch.__version__}'s layout")

    dtypes = live_dtypes()
    # `at::ScalarType` is a dense enum, so a gap means a code this torch uses is
    # named nowhere in its namespace -- and a tensor of that dtype would be
    # refused at launch as a layout mismatch.  Say so here rather than print a
    # stanza with a hole in it.
    holes = sorted(set(range(max(dtypes, default=0))) - set(dtypes))
    if holes:
        raise SystemExit(f"intj: torch {torch.__version__} names no dtype for codes {holes}")

    entries = " ".join(f"{code}={name}:{size}" for code, (name, size) in sorted(dtypes.items()))
    print(f"# torch {torch.__version__}, x86-64")
    print("[%d.%d]" % torch_version())
    print("layout: " + " ".join(f"{f}={getattr(layout, f)}" for f in _OFFSETS))
    print(textwrap.fill(entries, width=88, initial_indent="dtypes: ", subsequent_indent="  "))


if __name__ == "__main__":
    _main()
