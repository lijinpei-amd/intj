# pyright: standard
"""`benchmarks/bench_ffi_compare.py --sweep`, skipping unavailable access modes.

usage (from the intj root): python <this> --sweep [bench_ffi_compare args]

Runs the unmodified script's source with one change: the sweep's INTJ access
modes (STATIC_COMPILE, RUNTIME_SHIM) are filtered to those this torch supports.
On the MI350's torch 2.15 dev build RUNTIME_SHIM has no verified layout, and
the original sweep aborts on it.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd() / "benchmarks"))

from intj import TorchAccessMode  # noqa: E402
from intj.launcher import UnsupportedKernel, _resolve_torch_access_mode  # noqa: E402

LOOP = "for mode in (TorchAccessMode.STATIC_COMPILE, TorchAccessMode.RUNTIME_SHIM):"


def available():
    modes = []
    for mode in (TorchAccessMode.STATIC_COMPILE, TorchAccessMode.RUNTIME_SHIM):
        try:
            _resolve_torch_access_mode(mode, confirmed=True)
            modes.append(mode)
        except UnsupportedKernel as e:
            print(f"skipping {mode.value}: {str(e)[:120]}", file=sys.stderr)
    return tuple(modes)


path = pathlib.Path.cwd() / "benchmarks" / "bench_ffi_compare.py"
source = path.read_text()
assert LOOP in source, "bench_ffi_compare.py changed; update the wrapper"
source = source.replace(LOOP, "for mode in _AVAILABLE_MODES:")
namespace = {
    "__name__": "__main__",
    "__file__": str(path),
    "_AVAILABLE_MODES": available(),
}
sys.argv[0] = str(path)
exec(compile(source, str(path), "exec"), namespace)
