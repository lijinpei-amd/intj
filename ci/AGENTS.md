# ci/

`ninja -C ci <target>` is the only entry point; `.github/workflows/` call
targets and hold no build logic. Design: `docs/superpowers/specs/2026-10-12-ci-design.md`.

## Parameters

Ninja has no command-line variables, so scripts read the environment:

| variable | values | default |
|---|---|---|
| `INTJ_MODE` | `editable` (the checkout, via `PYTHONPATH`), `wheel` (the installed package) | `editable` |
| `PYTHON` | interpreter | `python3`; `/opt/venv/bin/python` in images |
| `CELL` | a `matrix.json` cell id | image targets only |
| `FLAVOR` | `cpu`, `rocm` | `cpu` |

## Targets

| target | does |
|---|---|
| `lint` | pre-commit over all files |
| `typecheck` | pyright |
| `wheel` | builds `ci/out/dist/intj-*.whl` (clears old ones) |
| `install` | editable: `pip install -e .`; wheel: (re)installs the one wheel in `ci/out/dist` |
| `test-cpu` | GPU-free tests |
| `test-gpu` | the full suite |
| `bench-host`, `bench-gpu` | `bench_launch.py`, output in `ci/out/bench/` |
| `matrix` | writes `ci/out/matrix.json` for the workflows |
| `image`, `push-image` | build / build and push the `$CELL`/`$FLAVOR` image |
| `check` (default) | `lint typecheck test-cpu` |

Ninja does not order command-line targets: run `wheel`, `install` and a test
target as separate invocations.

## Rules

- Every script starts with `. "$(dirname "$0")/env.sh"` and runs python via
  `py`, so the mode decides what `import intj` means. Wheel mode runs
  `$PYTHON -I` from `ci/out/run/` and refuses an intj outside site-packages.
- A new check is a script plus a `build` line, then a workflow step calling it.
- Anything that goes into an image is under `ci/dockerfiles/`: every file there
  feeds the tag hash.
