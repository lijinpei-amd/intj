# Every test that needs no GPU: the files below, minus tests marked `gpu`
# (ci/AGENTS.md). Without triton (the py3.8 and py3.9 cells) only the
# triton-free files run. test_kernel_cache.py stays out: its TSL/ABSL cases
# download their pinned trees, and abseil needs cmake.
. "$(dirname "$0")/env.sh"
run() { py -m pytest -q -p no:cacheprovider -m 'not gpu' "${@/#/$ROOT/tests/}"; }
# Apart: importing triton turns the GIL back on, and test_runtime.py checks
# that a free-threaded build keeps it off.
run test_runtime.py test_scalar_types.py
tests=(test_python_compat.py)
if py -c 'import importlib.util as u, sys; sys.exit(u.find_spec("triton") is None)' >/dev/null; then
  tests+=(test_annotation.py test_compat.py test_compile_options.py test_delayed.py
    test_grid.py test_invariant_host.py test_launcher_object.py test_tuning.py)
fi
run "${tests[@]}"
