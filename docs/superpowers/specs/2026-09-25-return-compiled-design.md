# Return CompiledKernel design

## Contract

make_launcher gains a keyword-only return_compiled: bool = False option,
independent of grid_arg, grid_cpp, and grid_py. The default launcher keeps
returning None. In true mode, a successful launch returns the actual Triton
CompiledKernel associated with the selected intj cache record. Repeated calls
using the same launcher module, device, and specialization return the same
Python object. Launch errors raise without returning an object.

return_compiled=True with no_gpu=True raises UnsupportedKernel during
make_launcher, before toolchain provisioning. In true mode, a zero-volume grid
still decodes arguments and compiles or finds the kernel, then returns it
without GPU dispatch. This matches Triton 3.7 and 3.8. The default zero-grid
path keeps returning None before specialization work.

The return flag is immutable render input in RenderContext and therefore
participates in ModuleKey.digest(). It does not enter the per-launch
specialization key: return mode does not alter which GPU binary is selected.

## Native ownership and hot path

The existing Python compile callback returns four launch fields in default
mode and those fields plus the CompiledKernel object in return mode. Its result
tuple keeps the object alive until native code chooses a cache record. After
the existing post-callback cache recheck, native code takes one strong
reference only when inserting a new record. If another invocation installed
the same key, intj uses that winning record and releases the losing callback
result. A failed insertion releases the newly acquired reference.

In true-mode generated modules, intj_kernel contains an owned PyObject*
compiled field. Every cache backend releases this reference when freeing a
record; the bound no-map singleton does the same. Module and bound GC traversal
visit cache-owned objects, and their clear hooks release them so cycles can
collect. Cache teardown marks the launcher closed before decref can run Python
finalizers. The current callback-owned kernel list remains in default mode;
true mode uses the record as owner.

A nonzero cache hit still packs arguments, hashes and looks up the key (or
reads the bound no-map singleton), validates the parameter count, and launches
through the cached GPU handle. It then returns Py_NewRef(record->compiled).
There is no Python callback, new object, or extra lookup on that hit. The
default generated module retains its existing record layout and return path.

## Integration and checks

Document the return contract in docs/Usage.md. Test a real GPU launch whose
returned CompiledKernel has TTIR, repeated-hit object identity, zero-grid
compile without dispatch, bound and no-map ownership, early no_gpu refusal,
module-key separation, callback replacement/reentry, and unchanged default
None results. Compile-check CUDA without claiming CUDA runtime coverage.

Retry Triton test launch sites blocked solely by needing this returned object.
Annotation parsing, Triton runtime hooks, descriptors, multi-CTA, PDL, and
other independent blockers remain outside this cycle and stay recorded in
the migration exception manifest. The migration preserves Python 3.10 and
Triton 3.7 support.

The compatibility bridge uses Triton's indexed launch while its compile-warmup
test context is active, so fake-pointer inputs are compiled without GPU
dispatch. CUDA TMA kernels that allocate global scratch remain on Triton.
