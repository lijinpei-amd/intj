"""Selects the kernel-cache backend and exposes its provisioning command.

Dependency installation lives in `intj.render.deps`. Run
`python -m intj.kernel_cache [--force] [tsl|absl|all]` to provision it ahead of
launcher checks and builds.
"""

from __future__ import annotations

import enum


class KernelCache(enum.Enum):
    """The hash map behind a module's kernel cache."""

    #: intj's own open-addressed table.  No dependency, and the only one that
    #: works in a module compiled as C.
    INTJ = "intj"
    #: `tsl::robin_map`, header-only.  Takes the precomputed hash on lookup.
    TSL = "tsl"
    #: `absl::flat_hash_map`.  Headers plus libraries, so it has to be built.
    ABSL = "absl"


def main(argv: list[str] | None = None) -> int:
    """Provisions the requested backends and returns the CLI exit status."""
    from .render import deps

    return deps.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
