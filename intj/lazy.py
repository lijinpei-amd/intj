"""Build and load `_intj_lazy`, the generic stub every launcher starts as.

Kernel-free, so it needs only the host C compiler and no GPU: `make_launcher`
allocates launchers from it at decoration time (see runtime/intj_lazy.c).
Cached on disk per interpreter and intj version; loaded by hand, never
imported, like every module intj builds.  No triton here: the Python-version
matrix (tests/run_python_matrix.sh) loads it on interpreters triton skips.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import sysconfig
import tempfile
import types
from pathlib import Path

from ._version import __version__

_RUNTIME = Path(__file__).parent / "runtime"
_SOURCE = _RUNTIME / "intj_lazy.c"
_HEADER = _RUNTIME / "intj_lazy.h"


def python_include() -> str:
    """This interpreter's `Python.h` directory; Debian's posix_local scheme included."""
    get_scheme = getattr(sysconfig, "get_default_scheme", None)  # 3.10+
    scheme = get_scheme() if get_scheme is not None else None
    if scheme == "posix_local":
        scheme = "posix_prefix"
    paths = sysconfig.get_paths(scheme=scheme) if scheme else sysconfig.get_paths()
    return paths["include"]


def _digest(cc: str) -> str:
    h = hashlib.sha256(_SOURCE.read_bytes() + _HEADER.read_bytes())
    h.update(
        repr(
            (__version__, sys.version, sysconfig.get_config_var("EXT_SUFFIX"), cc)
        ).encode()
    )
    return h.hexdigest()[:32]


def load_stub(root: Path, cc: str) -> types.ModuleType:
    """`_intj_lazy`, compiled into `root/<digest>/` on first use."""
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    so = root / _digest(cc) / f"_intj_lazy{suffix}"
    if not so.exists():
        so.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=so.parent) as staging:
            built = Path(staging) / so.name
            subprocess.check_call(
                [
                    cc,
                    "-O2",
                    "-shared",
                    "-fPIC",
                    f"-I{_RUNTIME}",
                    f"-I{python_include()}",
                    str(_SOURCE),
                    "-o",
                    str(built),
                ],
                stdout=subprocess.DEVNULL,
            )
            os.replace(built, so)  # atomic: a racing loader sees all of it or none
    spec = importlib.util.spec_from_file_location("_intj_lazy", so)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"intj: cannot load {so}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # pyright: ignore[reportAttributeAccessIssue]  # 3.8 stubs lack it
    return module
