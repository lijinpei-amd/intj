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
not of the host.

`create_launcher` provisions on demand, so the first use of a backend fetches it
(abseil also builds its 90 libraries: 28 s on 224 cores, minutes on a few).  It
says so on stderr first, because an implicit download is otherwise
indistinguishable from a hang.  To get it over with ahead of time, or on a
machine that will later be offline:

    python -m intj.kernel_cache tsl            # or absl, or all
    python -m intj.kernel_cache --force absl   # discard a half-written tree
"""

import contextlib
import dataclasses
import enum
import fcntl
import functools
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Generator
from pathlib import Path

#: What a failed `install` raises: network and disk (OSError, which
#: urllib.error.URLError subclasses), a bad tarball, a failed compiler, and the
#: RuntimeErrors raised here.  Narrow on purpose -- an internal bug should not
#: come back to the user as "the download failed, try again".
INSTALL_ERRORS = (OSError, RuntimeError, tarfile.TarError, subprocess.SubprocessError)

#: seconds; a stalled connection must not hang `create_launcher` forever
_DOWNLOAD_TIMEOUT = 60

#: written last, so a half-finished build tree never looks provisioned
_BUILT_MARKER = ".intj-built"

#: what the one-off abseil build costs, as the announce and the docs say it.
#: Measured: 28 s wall for all 90 libraries on 224 cores, and it scales with
#: them, so quote both ends rather than one machine's number.
_BUILD_COST = "~30 s on a big machine, minutes on a few cores"


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

    Only intj's own cache directory is consulted; `install` is what puts
    anything there.
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

    # the marker, not the archives: a build in flight has emitted some of its 90
    # libraries, and linking against that set fails with undefined absl symbols
    if not (root / "build" / _BUILT_MARKER).exists():
        return None
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
    tree is reused unless `force`, so the common call does no work and says
    nothing.  When there *is* work, it is announced -- a silent minutes-long
    build inside `create_launcher` is indistinguishable from a hang.

    One installer at a time, machine-wide: the 8 ranks of a torchrun job all
    miss together, and without the lock they would cmake into one build tree.
    """
    if cache is KernelCache.INTJ:
        return {"include_dirs": (), "library_dirs": (), "archives": ()}

    source = _SOURCES[cache]
    root = _source_root(cache)
    with _install_lock(cache):
        if force:
            shutil.rmtree(root, ignore_errors=True)
        if not (root / source.marker).exists():
            _announce(f"downloading {cache.value} {source.version} to {root}")
            _unpack(source, root)
        if source.builds and not (root / "build" / _BUILT_MARKER).exists():
            _announce(f"building {cache.value} {source.version} ({_BUILD_COST}, once)")
            _cmake_build(root)

    toolchain = toolchain_for(cache)
    if toolchain is None:  # pragma: no cover - a build that produced nothing
        raise RuntimeError(f"intj: {cache.value} is still unusable after installing it into {root}")
    return toolchain


def unavailable_message(cache: KernelCache, reason: object) -> str:
    """Why an on-demand install failed, and how to do it by hand.

    The reason goes last: a cmake failure brings a screenful of compiler
    diagnostics with it, and what the reader has to act on must not be buried
    above them.
    """
    source = _SOURCES[cache]
    what = "downloading and building" if source.builds else "downloading"
    python = Path(sys.executable).name
    return (
        f"intj: kernel_cache=KernelCache.{cache.name} needs {cache.value} {source.version} "
        f"in {_deps_root()}, and {what} it failed. Run "
        f"`{python} -m intj.kernel_cache {cache.value}` to retry on its own, add --force "
        f"to discard a half-written tree first, or use kernel_cache=KernelCache.INTJ, "
        f"which needs nothing. The failure was: {reason}"
    )


def _announce(message: str) -> None:
    print(f"intj: {message}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def _install_lock(cache: KernelCache) -> Generator[None]:
    """One installer per machine for `cache`, across processes.

    flock rather than a lock file we create and delete: the kernel drops it when
    the process dies, so a killed install does not wedge every later one.
    """
    deps = _deps_root()
    deps.mkdir(parents=True, exist_ok=True)
    with open(deps / f".{cache.value}.lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


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
        # timeout, because with none urlopen inherits socket's default of None:
        # a blackholed port 443 would hang create_launcher forever
        with urllib.request.urlopen(source.url, timeout=_DOWNLOAD_TIMEOUT) as response:  # noqa: S310 - pinned https URL
            blob = response.read()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != source.sha256:
            raise RuntimeError(
                f"intj: {source.url} hashed to {digest}, expected {source.sha256}"
            )
        archive.write_bytes(blob)
        with tarfile.open(archive) as tar:
            tar.extractall(Path(scratch) / "tree", filter="data")
        # one directory in, the way github archives are laid out
        (inner,) = (Path(scratch) / "tree").iterdir()
        # rename(2) onto a non-empty directory is ENOTEMPTY, so clear whatever an
        # interrupted install left: the caller already found no marker there.
        shutil.rmtree(root, ignore_errors=True)
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
    (build / _BUILT_MARKER).write_text("")  # last, and only on success


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in argv
    argv = [arg for arg in argv if arg != "--force"]
    wanted = [c for c in KernelCache if c is not KernelCache.INTJ]
    if argv and argv != ["all"]:
        try:
            wanted = [KernelCache(name) for name in argv]
        except ValueError:
            print("usage: python -m intj.kernel_cache [--force] [tsl|absl|all]", file=sys.stderr)
            return 2
    for cache in wanted:
        if cache is KernelCache.INTJ:
            continue
        # the same one line create_launcher would print, not a traceback: this
        # command is what that error tells the reader to run
        try:
            toolchain = install(cache, force=force)
        except INSTALL_ERRORS as error:
            print(unavailable_message(cache, error), file=sys.stderr)
            return 1
        print(f"intj: {cache.value} {_SOURCES[cache].version} in {_source_root(cache)}")
        print(f"  include {toolchain['include_dirs'][0]}")
        if toolchain["archives"]:
            print(f"  {len(toolchain['archives'])} libraries in {toolchain['library_dirs'][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
