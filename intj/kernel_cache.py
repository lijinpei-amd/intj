"""Which hash map a generated module uses for its kernel cache.

The map is on the launch path: it turns a spec key into a compiled kernel, once
per launch.  Measured on this workload -- key of N uint64 words, insert-only,
~100% hit, hash already computed by the caller -- the three are within a few ns
of each other, and `INTJ` wins at the sizes a kernel cache actually reaches:

    ns/lookup, 40-byte key         1 entry   8 entries   512 entries
    INTJ                              4.18        3.24          4.55
    TSL   (precalculated_hash)        3.88        3.93          5.59
    ABSL                              2.95       10.40         12.55

`ABSL` is fast at one entry only because its small-object path skips hashing
entirely; past that it re-computes the hash the caller already has, which no
abseil API lets you pass in.  `TSL` does take one.  They are here to be
measured, not because they are expected to win.

`TSL` and `ABSL` are **downloaded and built into intj's own cache directory**,
at a pinned version and checksum.  Nothing installed on the machine is searched
for or used: what a module was built against is then a property of intj's cache,
not of the host.  Provision with

    python -m intj.kernel_cache tsl      # or absl, or all
"""

import dataclasses
import enum
import functools
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


class KernelCache(enum.Enum):
    """The hash map behind a module's kernel cache."""

    #: intj's own open-addressed table.  No dependency, and the only one that
    #: works in a module compiled as C.
    INTJ = "intj"
    #: `tsl::robin_map`, header-only.  Takes the precomputed hash on lookup.
    TSL = "tsl"
    #: `absl::flat_hash_map`.  Headers plus libraries, so it has to be built.
    ABSL = "absl"


@dataclasses.dataclass(frozen=True)
class Dependency:
    """A pinned source release intj fetches for itself."""

    version: str
    url: str
    sha256: str
    #: proof the unpacked tree is what it claims, relative to its root
    marker: str
    #: where the headers are, relative to the root
    include: str
    #: built with cmake, rather than header-only
    builds: bool = False


_SOURCES: dict[KernelCache, Dependency] = {
    KernelCache.TSL: Dependency(
        version="1.3.0",
        url="https://github.com/Tessil/robin-map/archive/refs/tags/v1.3.0.tar.gz",
        sha256="a8424ad3b0affd4c57ed26f0f3d8a29604f0e1f2ef2089f497f614b1c94c7236",
        marker="include/tsl/robin_map.h",
        include="include",
    ),
    KernelCache.ABSL: Dependency(
        version="20250814.1",
        url="https://github.com/abseil/abseil-cpp/archive/refs/tags/20250814.1.tar.gz",
        sha256="1692f77d1739bacf3f94337188b78583cf09bab7e420d2dc6c5605a4f86785a1",
        marker="absl/container/flat_hash_map.h",
        include=".",
        builds=True,
    ),
}


def toolchain_for(cache: KernelCache) -> dict[str, tuple[str, ...]] | None:
    """Include and library directories for `cache`, or None if not provisioned.

    Only intj's own cache directory is consulted.  `install` is what puts
    anything there, and it is never called implicitly: a `create_launcher` that
    quietly reached for the network would be a surprise at an unpredictable
    moment.
    """
    if cache is KernelCache.INTJ:
        return {"include_dirs": (), "library_dirs": (), "archives": ()}

    source = _SOURCES[cache]
    root = _source_root(cache)
    if not (root / source.marker).exists():
        return None
    include = str((root / source.include).resolve())
    if not source.builds:
        return {"include_dirs": (include,), "library_dirs": (), "archives": ()}

    archives = _archives(root / "build")
    if not archives:
        return None
    return {
        "include_dirs": (include,),
        "library_dirs": (str(root / "build"),),
        "archives": archives,
    }


def install(cache: KernelCache, *, force: bool = False) -> dict[str, tuple[str, ...]]:
    """Download (and build, for abseil) `cache` into intj's cache directory.

    Returns the same toolchain `toolchain_for` does.  Idempotent: an existing
    tree is reused unless `force`.
    """
    if cache is KernelCache.INTJ:
        return {"include_dirs": (), "library_dirs": (), "archives": ()}

    source = _SOURCES[cache]
    root = _source_root(cache)
    if force:
        shutil.rmtree(root, ignore_errors=True)
    if not (root / source.marker).exists():
        _unpack(source, root)
    if source.builds and not _archives(root / "build"):
        _cmake_build(root)

    toolchain = toolchain_for(cache)
    if toolchain is None:  # pragma: no cover - a build that produced nothing
        raise RuntimeError(f"intj: {cache.value} is still unusable after installing it into {root}")
    return toolchain


def unavailable_message(cache: KernelCache) -> str:
    return (
        f"intj: kernel_cache=KernelCache.{cache.name} is not provisioned; run "
        f"`{Path(sys.executable).name} -m intj.kernel_cache {cache.value}` to download "
        f"{'and build ' if _SOURCES[cache].builds else ''}"
        f"{cache.value} {_SOURCES[cache].version} into {_deps_root()}"
    )


@functools.lru_cache(maxsize=1)
def _deps_root() -> Path:
    """`$TRITON_HOME/.triton/intj/deps`, beside the modules intj builds."""
    from triton import knobs

    return Path(knobs.cache.get_triton_dir("intj")) / "deps"


def _source_root(cache: KernelCache) -> Path:
    return _deps_root() / f"{cache.value}-{_SOURCES[cache].version}"


def _archives(directory: Path) -> tuple[str, ...]:
    """Every abseil library under `directory`.

    All of them, because abseil's own dependency order is not something intj
    should encode: the link line wraps them in `--start-group`.
    """
    if not directory.is_dir():
        return ()
    found = sorted(str(p) for p in directory.rglob("libabsl_*.a"))
    return tuple(found or sorted(str(p) for p in directory.rglob("libabsl_*.so")))


def _unpack(source: Dependency, root: Path) -> None:
    """Fetch, verify, extract -- then move into place, so a kill leaves nothing."""
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as scratch:
        archive = Path(scratch) / "source.tar.gz"
        with urllib.request.urlopen(source.url) as response:  # noqa: S310 - pinned https URL
            archive.write_bytes(response.read())
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != source.sha256:
            raise RuntimeError(
                f"intj: {source.url} hashed to {digest}, expected {source.sha256}"
            )
        with tarfile.open(archive) as tar:
            tar.extractall(Path(scratch) / "tree", filter="data")
        # one directory in, the way github archives are laid out
        (inner,) = (Path(scratch) / "tree").iterdir()
        if (root / source.marker).exists():  # a racing install won
            return
        os.replace(inner, root)


def _cmake_build(root: Path) -> None:
    build = root / "build"
    configure = [
        "cmake", "-S", str(root), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TESTING=OFF",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        # the modules that link this are built with -std=c++20; abseil mixes
        # standards badly, which is what ABSL_PROPAGATE_CXX_STD exists to stop
        "-DABSL_PROPAGATE_CXX_STD=ON",
        "-DCMAKE_CXX_STANDARD=20",
    ]
    for command in (configure, ["cmake", "--build", str(build), "-j", str(os.cpu_count() or 8)]):
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(
                f"intj: {' '.join(command[:2])} failed for {root.name}:\n{done.stderr[-2000:]}"
            )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    wanted = [c for c in KernelCache if c is not KernelCache.INTJ]
    if argv and argv != ["all"]:
        try:
            wanted = [KernelCache(name) for name in argv]
        except ValueError:
            print(f"usage: python -m intj.kernel_cache [tsl|absl|all]", file=sys.stderr)
            return 2
    for cache in wanted:
        if cache is KernelCache.INTJ:
            continue
        source = _SOURCES[cache]
        print(f"intj: installing {cache.value} {source.version} into {_source_root(cache)}")
        toolchain = install(cache)
        print(f"  include {toolchain['include_dirs'][0]}")
        if toolchain["archives"]:
            print(f"  {len(toolchain['archives'])} libraries in {toolchain['library_dirs'][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
