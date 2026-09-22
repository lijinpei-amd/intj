# pyright: standard
"""Build and run `bench_kernel_cache.cpp` for every kernel cache available here.

The C++ benchmark is the honest place to compare the backends: it drives
`intj_cache_*` directly, so nothing of a launch is in the number. This wrapper
builds one binary per backend (the implementation is a compile-time choice),
runs them, and prints one table.

Needs google/benchmark: `$INTJ_BENCHMARK_ROOT` pointing at a build tree, or an
installed copy. Everything is skipped without it -- including in CI, where the
numbers would be noise anyway.
"""

import json
import os
import pathlib
import subprocess
import sysconfig

import pytest

from intj.kernel_cache import KernelCache, toolchain_for

_RUNTIME = pathlib.Path(__import__("intj").__file__).parent / "runtime"
_SOURCE = pathlib.Path(__file__).parent / "bench_kernel_cache.cpp"
_NWORDS = 5  # a 3-tensor, 1-int kernel


def _benchmark_toolchain():
    """(include dir, [libraries]) for google/benchmark, or None."""
    root = os.environ.get("INTJ_BENCHMARK_ROOT")
    if root:
        include = pathlib.Path(root) / "include"
        libraries = sorted(str(p) for p in pathlib.Path(root).rglob("libbenchmark.a"))
        if include.is_dir() and libraries:
            return str(include), libraries[:1]
    for prefix in ("/usr/local", "/usr"):
        include = pathlib.Path(prefix) / "include"
        library = pathlib.Path(prefix) / "lib" / "libbenchmark.a"
        if (include / "benchmark" / "benchmark.h").exists() and library.exists():
            return str(include), [str(library)]
    return None


def _build(cache: KernelCache, include: str, libraries: list[str], out: pathlib.Path):
    toolchain = toolchain_for(cache)
    assert toolchain is not None
    command = [
        os.environ.get("CXX", "g++"), "-O3", "-DNDEBUG", "-std=c++20",
        f"-DINTJ_NWORDS={_NWORDS}",
        f"-DINTJ_CACHE_{cache.value.upper()}",
        f'-DINTJ_CACHE_NAME="{cache.value}"',
        str(_SOURCE), "-o", str(out),
        f"-I{_RUNTIME}", f"-I{include}", f"-I{sysconfig.get_paths()['include']}",
        *(f"-I{d}" for d in toolchain["include_dirs"]),
    ]
    if toolchain["archives"]:
        command += ["-Wl,--start-group", *toolchain["archives"], "-Wl,--end-group"]
    command += [*libraries, "-lpthread", f"-lpython{sysconfig.get_python_version()}"]
    command += [f"-L{sysconfig.get_config_var('LIBDIR')}"]
    return subprocess.run(command, capture_output=True, text=True)


def test_kernel_cache_benchmark(capsys):
    toolchain = _benchmark_toolchain()
    if toolchain is None:
        pytest.skip("google/benchmark not found; set $INTJ_BENCHMARK_ROOT")
    include, libraries = toolchain

    available = [c for c in KernelCache if toolchain_for(c) is not None]
    results: dict[str, dict[str, float]] = {}
    tmp = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "intj-cache-bench"
    tmp.mkdir(parents=True, exist_ok=True)

    for cache in available:
        binary = tmp / f"bench_{cache.value}"
        built = _build(cache, include, libraries, binary)
        assert built.returncode == 0, f"{cache.value} failed to build:\n{built.stderr[-2000:]}"
        run = subprocess.run(
            [str(binary), "--benchmark_format=json", "--benchmark_min_time=0.05s"],
            capture_output=True, text=True,
        )
        assert run.returncode == 0, f"{cache.value} failed to run:\n{run.stderr[-2000:]}"
        for entry in json.loads(run.stdout)["benchmarks"]:
            assert "error_message" not in entry, entry
            results.setdefault(entry["name"], {})[cache.value] = entry["real_time"]

    names = sorted(results, key=lambda n: (n.split("/")[0], int(n.split("/")[-1]) if "/" in n else 0))
    with capsys.disabled():
        print(f"\nkernel cache, {_NWORDS}-word key, ns per operation\n")
        print(f"{'':<16}" + "".join(f"{c.value:>10}" for c in available))
        for name in names:
            row = "".join(f"{results[name].get(c.value, float('nan')):>10.2f}" for c in available)
            print(f"{name:<16}{row}")

    # every backend must find what it stored, at a sane speed
    for name, row in results.items():
        for backend, ns in row.items():
            assert 0.0 < ns < 1000.0, f"{backend} {name} = {ns} ns"
