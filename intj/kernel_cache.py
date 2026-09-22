"""Which hash map a generated module uses for its kernel cache.

The map is on the launch path: it turns a spec key into a compiled kernel, once
per launch.  Measured on this workload -- key of N uint64 words, insert-only,
~100% hit, hash already computed by the caller -- the three are within a few ns
of each other, and `INTJ` wins at the sizes a kernel cache actually reaches:

    ns/lookup, 40-byte key         1 entry   8 entries   512 entries
    INTJ                              1.93        1.95          2.53
    TSL   (precalculated_hash)        2.18        2.36          3.13
    ABSL                              2.15        9.63         10.50

`ABSL` is fast at one entry only because its small-object path skips hashing
entirely; past that it re-computes the hash the caller already has, which no
abseil API lets you pass in.  `TSL` does take one.  They are here to be
measured, not because they are expected to win.
"""

import enum
import functools
import os
from pathlib import Path


class KernelCache(enum.Enum):
    """The hash map behind a module's kernel cache."""

    #: intj's own open-addressed table.  No dependency, and the only one that
    #: works in a module compiled as C.
    INTJ = "intj"
    #: `tsl::robin_map`, header-only.  Takes the precomputed hash on lookup.
    TSL = "tsl"
    #: `absl::flat_hash_map`.  Needs abseil's headers *and* its libraries.
    ABSL = "absl"


def toolchain_for(cache: KernelCache) -> dict[str, tuple[str, ...]] | None:
    """Include and library directories for `cache`, or None if it is unavailable.

    `TSL` and `ABSL` are found through `$INTJ_TSL_INCLUDE`, `$INTJ_ABSL_INCLUDE`
    and `$INTJ_ABSL_LIB`, falling back to the usual system directories.  Neither
    is vendored: they are options to measure, so the build points at whichever
    copy the caller already has.
    """
    if cache is KernelCache.INTJ:
        return {"include_dirs": (), "library_dirs": (), "archives": ()}
    if cache is KernelCache.TSL:
        include = _find_include("INTJ_TSL_INCLUDE", "tsl/robin_map.h")
        if include is None:
            return None
        return {"include_dirs": (include,), "library_dirs": (), "archives": ()}

    include = _find_include("INTJ_ABSL_INCLUDE", "absl/container/flat_hash_map.h")
    library = _find_absl_libraries()
    if include is None or library is None:
        return None
    directory, archives = library
    return {"include_dirs": (include,), "library_dirs": (directory,), "archives": archives}


def unavailable_message(cache: KernelCache) -> str:
    if cache is KernelCache.TSL:
        return (
            "intj: kernel_cache=KernelCache.TSL needs tsl/robin_map.h; set "
            "$INTJ_TSL_INCLUDE to the directory containing `tsl/`"
        )
    return (
        "intj: kernel_cache=KernelCache.ABSL needs abseil's headers and libraries; set "
        "$INTJ_ABSL_INCLUDE to the directory containing `absl/` and $INTJ_ABSL_LIB to "
        "the one containing libabsl_*.a or .so"
    )


_SYSTEM_INCLUDES = ("/usr/local/include", "/usr/include")
_SYSTEM_LIBRARIES = ("/usr/local/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib")


@functools.lru_cache(maxsize=None)
def _find_include(variable: str, header: str) -> str | None:
    candidates = [os.environ[variable]] if variable in os.environ else list(_SYSTEM_INCLUDES)
    for directory in candidates:
        if (Path(directory) / header).exists():
            return directory
    return None


@functools.lru_cache(maxsize=1)
def _find_absl_libraries() -> tuple[str, tuple[str, ...]] | None:
    """The directory holding libabsl_*, and every archive in it.

    Every archive, because abseil's own dependency order is not something intj
    should encode: the link line wraps them in `--start-group`.  A shared-library
    install lands here too -- the `.so`s are passed the same way.
    """
    roots = [os.environ["INTJ_ABSL_LIB"]] if "INTJ_ABSL_LIB" in os.environ else list(_SYSTEM_LIBRARIES)
    for root in roots:
        directory = Path(root)
        if not directory.is_dir():
            continue
        archives = sorted(str(p) for p in directory.rglob("libabsl_*.a"))
        archives += sorted(str(p) for p in directory.rglob("libabsl_*.so"))
        if archives:
            return (str(directory), tuple(archives))
    return None
