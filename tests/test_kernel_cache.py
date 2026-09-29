# pyright: standard
"""Build and run `bench_kernel_cache.cpp` for every kernel cache available here.

The C++ benchmark is the honest place to compare the backends: it drives
`intj_cache_*` directly, so nothing of a launch is in the number. This wrapper
builds one binary per backend (the implementation is a compile-time choice),
runs them, and prints one table.

The backends come from intj's own cache directory (`python -m intj.kernel_cache
all`), so whichever are provisioned get measured. google/benchmark itself is the
one thing still taken from the environment: `$INTJ_BENCHMARK_ROOT` pointing at a
build tree, or an installed copy. Everything is skipped without it.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sysconfig

import pytest
import triton
import triton.language as tl

from intj import Constexpr, TorchAccessMode, make_launcher
from intj.kernel_cache import KernelCache, toolchain_for
from intj.launcher import module_of, override_compile
from intj.python_intf import cpython_abi

_RUNTIME = pathlib.Path(__import__("intj").__file__).parent / "runtime"
_PYTHON_INTF = _RUNTIME.parent / "python_intf"
_SOURCE = pathlib.Path(__file__).parent / "bench_kernel_cache.cpp"
#: The three key lengths the packed layout actually produces, which are also the
#: three hash specializations: 1 = no constexpr and at most 7 params, the
#: splitmix64 finalizer; 2 =
#: add_kernel(x, y, out, n, BLOCK_SIZE); 5 = six tensors, three ints, three
#: constexpr.  All three have to compile and self-check, not just the one a
#: given kernel happens to hit.
_NWORDS = (1, 2, 5)


def _benchmark_toolchain():
    """Returns `(include dir, [libraries])` for google/benchmark, or None if absent."""
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


def _build(
    cache: KernelCache,
    nwords: int,
    include: str,
    libraries: list[str],
    out: pathlib.Path,
):
    """Compiles the C++ benchmark for `cache` at key length `nwords` into `out`.

    Returns:
        The finished compiler process, with its output captured as text.
    """
    toolchain = toolchain_for(cache)
    assert toolchain is not None
    command = [
        os.environ.get("CXX", "g++"),
        "-O3",
        # rows are 1-3 ns: without it an unrelated edit moves a miss row by
        # up to 1 ns through loop placement alone (2026-09-27 regression report)
        "-falign-loops=64",
        "-DNDEBUG",
        "-std=c++20",
        f"-DINTJ_NWORDS={nwords}",
        f"-DINTJ_CACHE_{cache.value.upper()}",
        f'-DINTJ_CACHE_NAME="{cache.value}"',
        str(_SOURCE),
        "-o",
        str(out),
        f'-DINTJ_CPYTHON_STATIC_COMPILE_HEADER="{cpython_abi.header_for()}"',
        f"-I{_PYTHON_INTF}",
        f"-I{_RUNTIME}",
        f"-I{include}",
        f"-I{sysconfig.get_paths()['include']}",
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
    # keyed on (nwords, name): every binary emits the same benchmark names, so a
    # narrower key would let the last key length overwrite the others in silence
    results: dict[tuple[int, str], dict[str, float]] = {}
    tmp = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "intj-cache-bench"
    tmp.mkdir(parents=True, exist_ok=True)

    for nwords in _NWORDS:
        for cache in available:
            binary = tmp / f"bench_{cache.value}_{nwords}"
            built = _build(cache, nwords, include, libraries, binary)
            assert built.returncode == 0, (
                f"{cache.value} at {nwords} words failed to build:\n{built.stderr[-2000:]}"
            )
            run = subprocess.run(
                [str(binary), "--benchmark_format=json", "--benchmark_min_time=0.05s"],
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, (
                f"{cache.value} at {nwords} words failed to run:\n{run.stderr[-2000:]}"
            )
            for entry in json.loads(run.stdout)["benchmarks"]:
                assert "error_message" not in entry, entry
                results.setdefault((nwords, entry["name"]), {})[cache.value] = entry[
                    "real_time"
                ]

    with capsys.disabled():
        for nwords in _NWORDS:
            names = sorted(
                (n for w, n in results if w == nwords),
                key=lambda n: (
                    n.split("/")[0],
                    int(n.split("/")[-1]) if "/" in n else 0,
                ),
            )
            print(f"\nkernel cache, {nwords}-word key, ns per operation\n")
            print(f"{'':<16}" + "".join(f"{c.value:>10}" for c in available))
            for name in names:
                row = "".join(
                    f"{results[nwords, name].get(c.value, float('nan')):>10.2f}"
                    for c in available
                )
                print(f"{name:<16}{row}")

    # every backend must find what it stored, at a sane speed
    for (nwords, name), row in results.items():
        for backend, ns in row.items():
            assert 0.0 < ns < 1000.0, f"{backend} {name} at {nwords} words = {ns} ns"


@triton.jit
def keyed_store(x, K: tl.constexpr):
    tl.store(x, K)


def test_block_dim_zero_is_refused():
    """block_dim 0 marks an empty slot in intj's map; storing one would hide it."""
    launch = make_launcher(
        keyed_store, no_gpu=True, torch_access_mode=TorchAccessMode.INTERPRETER
    )
    override_compile(module_of(launch), lambda key, nparams, *_: (0, 0, 0, nparams))
    with pytest.raises(ValueError, match="block_dim 0"):
        launch(0, 0, 1, 0, 1)


@pytest.mark.parametrize("cache", list(KernelCache))
def test_records_survive_rehash(cache):
    """Records live in the slots now, so every growth moves them."""
    launch = make_launcher(
        keyed_store,
        no_gpu=True,
        kernel_cache=cache,
        torch_access_mode=TorchAccessMode.INTERPRETER,
    )
    compiled = []

    def compile_key(key, nparams, device, x, k):
        compiled.append(k)
        return 0, 1, 0, nparams

    override_compile(module_of(launch), compile_key)
    for _ in range(2):
        for k in range(100):
            launch(0, 0, 1, 0, k)
    assert compiled == list(range(100))


@pytest.mark.parametrize("cache", list(KernelCache))
def test_one_word_map_survives_rehash(cache):
    """INTJ_NWORDS 1 is its own map instantiation; grow it past its 8 slots."""
    launch = make_launcher(
        keyed_store,
        no_gpu=True,
        kernel_cache=cache,
        torch_access_mode=TorchAccessMode.INTERPRETER,
        extra_annotation={"K": Constexpr(type=tl.int8)},
    )
    so = pathlib.Path(str(module_of(launch).__file__))
    stem = so.name.split(".")[0]
    (source,) = [
        so.with_name(stem + ext)
        for ext in (".c", ".cpp")
        if so.with_name(stem + ext).exists()
    ]
    assert "#define INTJ_NWORDS 1\n" in source.read_text()
    compiled = []

    def compile_key(key, nparams, device, x, k):
        compiled.append(k)
        return 0, 1, 0, nparams

    override_compile(module_of(launch), compile_key)
    for _ in range(2):
        for k in range(20):
            launch(0, 0, 1, 0, k)
    assert compiled == list(range(20))
