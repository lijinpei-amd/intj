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
| `TORCH` | a torch version for `install` to put in first, from `$INTJ_TORCH_INDEX` | the venv's torch |
| `INTJ_TEST_TORCH_MODES` | Torch access modes `tests/test_runtime.py` must run, comma-separated | every available one |

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
| `matrix` | writes `ci/out/matrix.json` for the workflows, with the cells not yet published |
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
- A published tag is never rebuilt; to refresh base images and unpinned
  packages, change any file in `ci/dockerfiles/` (e.g. bump a comment). The
  workflows start an image job only for unpublished tags.
- `matrix.json`'s `torch` pins one version per minor in
  `intj/torch_intf/torch_abi.toml`, with the pythons that have a CPU wheel for
  it; `image.py matrix` refuses a mismatch. Adding a toml row means adding a
  pin. 2.9.1 leaves out its free-threaded wheels: from several threads,
  `tensor.untyped_storage()` crashes it in `PyObject_ClearWeakRefs`, without
  intj (2.10 is fine).

## Known gaps

- `gpu.yml` and the rocm images are untested until a `[self-hosted, gfx942]`
  runner is registered. CUDA has no images (no NVIDIA GPU to test on).
- The self-hosted runner must be Actions Runner >= 2.327.1 (the actions run on
  Node 24) and provide `docker`, `ninja` and `python3`
  (`gpu.yml` runs `image.py` and `ninja` directly on the host, outside a
  container).
