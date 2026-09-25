# pyright: standard
"""Port TVM FFI's CUDA CUBIN overhead example to ROCm HSACO.

Compile one Triton empty kernel, load its HSACO through TVM FFI and HIP,
then time that same kernel through TVM FFI, Triton, and INTJ. Run from the
INTJ repository root with PYTHONPATH=$PWD; requires a ROCm GPU and tvm-ffi.
"""

import argparse
import hashlib
import statistics
from typing import Any

import torch
import triton
import triton.language as tl
import tvm_ffi
import tvm_ffi.cpp

from intj import make_launcher
from bench_launch import bench


@triton.jit
def empty_kernel(a, b, c, n, BLOCK: tl.constexpr):
    pass


HIP_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/string.h>

static hipModule_t module = nullptr;
static hipFunction_t kernel = nullptr;
static unsigned threads = 0;

void load_hsaco(const tvm::ffi::Bytes& image, int block_threads) {
  TVM_FFI_CHECK(!module && block_threads > 0, ValueError) << "invalid HIP module or block";
  hipError_t err = hipModuleLoadData(&module, image.data());
  TVM_FFI_CHECK(err == hipSuccess, RuntimeError) << hipGetErrorString(err);
  err = hipModuleGetFunction(&kernel, module, "empty_kernel");
  if (err != hipSuccess) {
    hipModuleUnload(module);
    module = nullptr;
    TVM_FFI_CHECK(false, RuntimeError) << hipGetErrorString(err);
  }
  threads = block_threads;
}

void launch_empty(tvm::ffi::TensorView a, tvm::ffi::TensorView b,
                  tvm::ffi::TensorView c) {
  TVM_FFI_CHECK(kernel, RuntimeError) << "load_hsaco must run first";
  TVM_FFI_CHECK(a.ndim() == 1 && b.ndim() == 1 && c.ndim() == 1, ValueError)
      << "expected 1D tensors";
  uint32_t n = static_cast<uint32_t>(a.size(0));
  void* ap = a.data_ptr();
  void* bp = b.data_ptr();
  void* cp = c.data_ptr();
  uint64_t scratch = 0;
  void* args[] = {&ap, &bp, &cp, &n, &scratch, &scratch};
  DLDevice device = a.device();
  hipStream_t stream = static_cast<hipStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));
  hipError_t err = hipModuleLaunchKernel(kernel, 1, 1, 1, threads, 1, 1,
                                        0, stream, args, nullptr);
  TVM_FFI_CHECK(err == hipSuccess, RuntimeError) << hipGetErrorString(err);
}
"""


def main(iters: int, batches: int) -> None:
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("a ROCm GPU is required")
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    tensors = tuple(torch.empty(128, device="cuda") for _ in range(3))
    ffi_args = tuple(tvm_ffi.from_dlpack(tensor) for tensor in tensors)
    native: Any = empty_kernel[(1,)]
    compiled = native(tensors[0], tensors[1], tensors[2], 128, 128)
    hsaco = compiled.kernel
    assert hsaco == compiled.asm["hsaco"] and hsaco.startswith(b"\x7fELF")
    assert compiled.name == "empty_kernel" and compiled.metadata.shared == 0
    block_threads = compiled.metadata.num_warps * compiled.metadata.warp_size
    mod = tvm_ffi.cpp.load_inline(
        name="intj_hip_hsaco_overhead",
        cuda_sources=HIP_SOURCE,
        functions=["load_hsaco", "launch_empty"],
    )
    mod.load_hsaco(hsaco, block_threads)
    bound = make_launcher(empty_kernel, bind_device=True).bind_device(device)
    callback_count = 0

    def compiled_function(_key, nparams, ordinal, *_args):
        nonlocal callback_count
        callback_count += 1
        assert ordinal == device and nparams == 4
        return compiled.function, block_threads, compiled.metadata.shared, nparams

    bound.__self__.__self__.set_compile_callback(compiled_function)
    cases = [
        ("TVM FFI HIP HSACO", mod.launch_empty, ffi_args),
        ("Triton same HSACO", native, (*tensors, 128, 128)),
        ("INTJ same function", bound, (stream, (1,), *tensors, 128, 128)),
    ]
    results = {name: [] for name, *_ in cases}
    with tvm_ffi.use_raw_stream(ffi_args[0].device, stream):
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
        f"hsaco_sha256={hashlib.sha256(hsaco).hexdigest()} kernel={compiled.name} "
        f"threads={block_threads} shared={compiled.metadata.shared} callback_count={callback_count}"
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
