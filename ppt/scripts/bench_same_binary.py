# pyright: standard
"""One Triton binary, three launchers: TVM FFI, Triton and intj, for a 1-argument kernel.

usage (from the intj root, PYTHONPATH=$PWD): python <this> [--iters N] [--batches N]

`benchmarks/bench_hip_module_launch.py` with a kernel of one tensor argument, on
both backends. Triton compiles `empty_kernel_1(a)` once. TVM FFI loads that same
binary: on ROCm through hipModuleLoadData, on CUDA through TVM FFI's own cubin
launcher (as in its examples/cubin_launcher/benchmark_overhead.py). intj launches
Triton's loaded function through `override_compile`. Times host enqueue, no timed
sync; batches alternate the case order.
"""

import argparse
import hashlib
import pathlib
import statistics
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path.cwd() / "benchmarks"))

import torch  # noqa: E402
import triton  # noqa: E402
import tvm_ffi  # noqa: E402  # pyright: ignore[reportMissingImports]
import tvm_ffi.cpp  # noqa: E402  # pyright: ignore[reportMissingImports]

from bench_launch import bench  # noqa: E402  # pyright: ignore[reportMissingImports]  # benchmarks/ joins sys.path above
from intj import make_launcher  # noqa: E402
from intj.launcher import module_of, override_compile  # noqa: E402


@triton.jit
def empty_kernel_1(a):
    pass


# Triton appends two scratch pointers (global, profile) to every kernel's parameters.
HIP_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/string.h>

static hipModule_t module = nullptr;
static hipFunction_t kernel = nullptr;
static unsigned threads = 0;

void load_binary(const tvm::ffi::Bytes& image, int block_threads) {
  TVM_FFI_CHECK(!module && block_threads > 0, ValueError) << "invalid HIP module or block";
  hipError_t err = hipModuleLoadData(&module, image.data());
  TVM_FFI_CHECK(err == hipSuccess, RuntimeError) << hipGetErrorString(err);
  err = hipModuleGetFunction(&kernel, module, "empty_kernel_1");
  TVM_FFI_CHECK(err == hipSuccess, RuntimeError) << hipGetErrorString(err);
  threads = block_threads;
}

void launch_empty(tvm::ffi::TensorView a) {
  TVM_FFI_CHECK(kernel, RuntimeError) << "load_binary must run first";
  TVM_FFI_CHECK(a.ndim() == 1, ValueError) << "expected a 1D tensor";
  void* ap = a.data_ptr();
  uint64_t scratch = 0;
  void* args[] = {&ap, &scratch, &scratch};
  DLDevice device = a.device();
  hipStream_t stream = static_cast<hipStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));
  hipError_t err = hipModuleLaunchKernel(kernel, 1, 1, 1, threads, 1, 1,
                                        0, stream, args, nullptr);
  TVM_FFI_CHECK(err == hipSuccess, RuntimeError) << hipGetErrorString(err);
}
"""

CUDA_SOURCE = r"""
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

TVM_FFI_EMBED_CUBIN(empty_cubin);

void launch_empty(tvm::ffi::TensorView a) {
  static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(empty_cubin, "empty_kernel_1");
  TVM_FFI_CHECK(a.ndim() == 1, ValueError) << "expected a 1D tensor";
  void* ap = a.data_ptr();
  uint64_t scratch = 0;
  void* args[] = {&ap, &scratch, &scratch};
  tvm::ffi::dim3 grid(1);
  tvm::ffi::dim3 block(THREADS);
  DLDevice device = a.device();
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(kernel.Launch(args, grid, block, stream));
}
"""


def main(iters: int, batches: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("a GPU is required")
    hip = torch.version.hip is not None
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    tensor = torch.empty(128, device="cuda")
    ffi_arg = tvm_ffi.from_dlpack(tensor)
    native: Any = empty_kernel_1[(1,)]
    compiled = native(tensor)
    binary = compiled.kernel
    assert binary.startswith(b"\x7fELF")
    assert compiled.name == "empty_kernel_1" and compiled.metadata.shared == 0
    threads = compiled.metadata.num_warps * compiled.metadata.warp_size
    if hip:
        mod = tvm_ffi.cpp.load_inline(
            name="intj_same_binary_hip",
            cuda_sources=HIP_SOURCE,
            functions=["load_binary", "launch_empty"],
        )
        mod.load_binary(binary, threads)
    else:
        mod = tvm_ffi.cpp.load_inline(
            name="intj_same_binary_cuda",
            cuda_sources=CUDA_SOURCE.replace("THREADS", str(threads)),
            functions=["launch_empty"],
            embed_cubin={"empty_cubin": binary},
        )
    bound = make_launcher(empty_kernel_1).__self__.bind_device(device)
    callback_count = 0

    def compiled_function(_key, nparams, ordinal, *_args):
        nonlocal callback_count
        callback_count += 1
        assert ordinal == device and nparams == 1
        return compiled.function, threads, compiled.metadata.shared, nparams

    override_compile(module_of(bound), compiled_function)
    cases = [
        ("TVM FFI same binary", mod.launch_empty, (ffi_arg,)),
        ("Triton same binary", native, (tensor,)),
        ("INTJ same binary", bound, (stream, (1,), tensor)),
    ]
    results = {name: [] for name, *_ in cases}
    with tvm_ffi.use_raw_stream(ffi_arg.device, stream):
        for _, fn, args in cases:
            fn(*args)
        torch.cuda.synchronize()
        assert callback_count == 1
        for sample in range(batches):
            for name, fn, args in reversed(cases) if sample % 2 else cases:
                results[name].append(
                    bench(fn, args, iters, 1, torch.cuda.synchronize) / 1000
                )
    print(
        f"binary_sha256={hashlib.sha256(binary).hexdigest()} kind={'hsaco' if hip else 'cubin'} "
        f"kernel={compiled.name} threads={threads} shared={compiled.metadata.shared} "
        f"callback_count={callback_count}"
    )
    print(f"iters={iters} batches={batches} unit=us/call (host enqueue, no timed sync)")
    for name, samples in results.items():
        print(f"{name:<25} median={statistics.median(samples):.4f} samples={samples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batches", type=int, default=9)
    options = parser.parse_args()
    if options.iters < 1 or options.batches < 1:
        parser.error("--iters and --batches must be positive")
    main(options.iters, options.batches)
