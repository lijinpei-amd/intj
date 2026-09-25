# pyright: standard
"""Compare cached INTJ launches with preconverted TVM FFI calls.

Requires apache-tvm-ffi in the same environment. Run from the INTJ root with
PYTHONPATH=. /path/to/venv/bin/python benchmarks/bench_ffi_compare.py [--iters N]
[--batches N]. All timings are host call/enqueue times, excluding synchronization.
Use --sweep [--counts 0 3 5 8 16 32 64] for static-compile/runtime-shim and preconverted FFI calls.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import statistics
import tempfile

import torch
import triton
import triton.language as tl
import tvm_ffi
import tvm_ffi.cpp

from intj import Argument, NEVER, TorchAccessMode, make_launcher
from bench_launch import bench


@triton.jit
def empty_kernel(x, y, z):
    pass


@triton.jit
def mixed_kernel(x, y, z, n, a):
    if n > 0:
        tl.store(z, tl.load(x) + tl.load(y) * a)


GPU_PREAMBLE = r"""
#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
using GpuStream = hipStream_t;
constexpr int kMixedThreads = 4 * 64;
#else
#include <cuda_runtime.h>
using GpuStream = cudaStream_t;
constexpr int kMixedThreads = 4 * 32;
#endif
"""

GPU_SOURCE = GPU_PREAMBLE + r"""
__global__ void EmptyKernel(const float*, const float*, float*) {}

__global__ void MixedKernel(const float* x, const float* y, float* z, int n, float a) {
  if (threadIdx.x == 0 && n > 0) z[0] = x[0] + y[0] * a;
}

void typed_nop(tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView) {}

void typed_nop_mixed(tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView,
                     int, float) {}

void launch_empty(tvm::ffi::TensorView x, tvm::ffi::TensorView y, tvm::ffi::TensorView z) {
  GpuStream stream = static_cast<GpuStream>(
      TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
  EmptyKernel<<<1, 1, 0, stream>>>(static_cast<const float*>(x.data_ptr()),
                                    static_cast<const float*>(y.data_ptr()),
                                    static_cast<float*>(z.data_ptr()));
}

void launch_mixed(tvm::ffi::TensorView x, tvm::ffi::TensorView y, tvm::ffi::TensorView z,
                  int n, float a) {
  GpuStream stream = static_cast<GpuStream>(
      TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
  MixedKernel<<<1, kMixedThreads, 0, stream>>>(static_cast<const float*>(x.data_ptr()),
                                                static_cast<const float*>(y.data_ptr()),
                                                static_cast<float*>(z.data_ptr()), n, a);
}
"""


def sweep_kinds(count: int) -> tuple[str, ...]:
    """Use three tensors, then repeat int, float, tensor."""
    return tuple("tensor" if i < 3 else ("int", "float", "tensor")[(i - 3) % 3]
                 for i in range(count))


def sweep_source(counts: list[int], device: int) -> str:
    source = [GPU_PREAMBLE, r"""
#if defined(__HIP_PLATFORM_AMD__)
constexpr int kSweepDeviceType = kDLROCM;
#else
constexpr int kSweepDeviceType = kDLCUDA;
#endif
"""]
    ffi_types = {"tensor": "tvm::ffi::TensorView", "int": "int", "float": "float"}
    gpu_types = {"tensor": "const float*", "int": "int", "float": "float"}
    for count in counts:
        kinds = sweep_kinds(count)
        ffi_params = ", ".join(f"{ffi_types[kind]} a{i}" for i, kind in enumerate(kinds))
        gpu_params = ", ".join(f"{gpu_types[kind]} a{i}" for i, kind in enumerate(kinds))
        gpu_args = ", ".join(
            f"static_cast<const float*>(a{i}.data_ptr())" if kind == "tensor" else f"a{i}"
            for i, kind in enumerate(kinds)
        )
        stream_device = ("a0.device().device_type, a0.device().device_id" if count
                         else f"kSweepDeviceType, {device}")
        source.append(
            f"\n__global__ void SweepKernel_{count}({gpu_params}) {{}}\n"
            f"void typed_{count}({ffi_params}) {{}}\n"
            f"void launch_{count}({ffi_params}) {{\n"
            f"  GpuStream stream = static_cast<GpuStream>("
            f"TVMFFIEnvGetStream({stream_device}));\n"
            f"  SweepKernel_{count}<<<1, kMixedThreads, 0, stream>>>({gpu_args});\n"
            "}\n"
        )
    return "".join(source)


def sweep(counts: list[int], iters: int, batches: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA or ROCm GPU is required")
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    tensor = torch.empty(2, device="cuda", dtype=torch.float32)
    converted = tvm_ffi.from_dlpack(tensor)
    mod = tvm_ffi.cpp.load_inline(
        name="benchmark_ffi_arg_sweep", cuda_sources=sweep_source(counts, device),
        functions=[name for count in counts for name in (f"typed_{count}", f"launch_{count}")],
    )
    packed_nop = tvm_ffi.get_global_func("testing.nop")
    cases = []
    with tempfile.TemporaryDirectory() as tmp:
        for count in counts:
            kinds = sweep_kinds(count)
            names = ", ".join(f"a{i}" for i in range(count))
            path = pathlib.Path(tmp) / f"sweep_{count}.py"
            path.write_text(
                f"import triton\n\n@triton.jit\ndef sweep_kernel_{count}({names}):\n    pass\n"
            )
            spec = importlib.util.spec_from_file_location(path.stem, path)
            assert spec is not None and spec.loader is not None
            kernel_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(kernel_module)  # pyright: ignore[reportAttributeAccessIssue]  # 3.8 stubs lack it
            kernel = getattr(kernel_module, f"sweep_kernel_{count}")
            annotation = {
                f"a{i}": Argument(
                    type={"tensor": tl.pointer_type(tl.float32), "int": tl.int32,
                          "float": tl.float32}[kind],
                    specialize=NEVER,
                ) for i, kind in enumerate(kinds)
            }
            args = tuple({"tensor": tensor, "int": 17, "float": 1.25}[kind]
                         for kind in kinds)
            ffi_args = tuple({"tensor": converted, "int": 17, "float": 1.25}[kind]
                             for kind in kinds)
            cases.extend((
                (count, "FFI packed nop", packed_nop, ffi_args),
                (count, "FFI typed nop", getattr(mod, f"typed_{count}"), ffi_args),
                (count, "FFI empty kernel", getattr(mod, f"launch_{count}"), ffi_args),
            ))
            for mode in (TorchAccessMode.STATIC_COMPILE, TorchAccessMode.RUNTIME_SHIM):
                launch = make_launcher(kernel, extra_annotation=annotation,
                                       bind_device=True, torch_access_mode=mode).bind_device(device)
                assert launch.__self__.__self__.spec_key(*args) == (b"", count)
                cases.append((count, f"INTJ {mode.value} kernel", launch, (stream, (1,), *args)))

        results: dict[tuple[int, str], list[float]] = {
            (count, name): [] for count, name, *_ in cases
        }
        with tvm_ffi.use_raw_stream(converted.device, stream):
            for sample in range(batches):
                for count, name, fn, args in reversed(cases) if sample % 2 else cases:
                    results[count, name].append(
                        bench(fn, args, iters, 1, torch.cuda.synchronize) / 1_000
                    )
        print(f"counts={counts} iters={iters} batches={batches} unit=us/call "
              "(host-only, no timed sync); INTJ device-bound, no specialization key")
        for (count, name), samples in results.items():
            print(f"args={count:2d} {name:<24} median={statistics.median(samples):.4f} "
                  f"samples={samples}")


def main(iters: int, batches: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA or ROCm GPU is required")

    mod = tvm_ffi.cpp.load_inline(
        name="benchmark_ffi_nop_launch", cuda_sources=GPU_SOURCE,
        functions=["typed_nop", "launch_empty", "typed_nop_mixed", "launch_mixed"],
    )
    tensors = (
        torch.full((2,), 2.0, device="cuda"),
        torch.full((2,), 3.0, device="cuda"),
        torch.empty(2, device="cuda"),
    )
    converted = tuple(tvm_ffi.from_dlpack(tensor) for tensor in tensors)
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream().cuda_stream
    grid = (1,)
    packed_nop = tvm_ffi.get_global_func("testing.nop")
    intj_launch = make_launcher(empty_kernel)
    intj_bound = make_launcher(empty_kernel, bind_device=True).bind_device(device)
    annotation = {
        **{name: Argument(type=tl.pointer_type(tl.float32), specialize=NEVER)
           for name in ("x", "y", "z")},
        "n": Argument(type=tl.int32, specialize=NEVER),
        "a": Argument(type=tl.float32, specialize=NEVER),
    }
    intj_mixed = make_launcher(mixed_kernel, extra_annotation=annotation)
    intj_mixed_bound = make_launcher(
        mixed_kernel, extra_annotation=annotation, bind_device=True
    ).bind_device(device)
    mixed_args = (*tensors, 1, 1.5)
    mixed_converted = (*converted, 1, 1.5)
    other_args = (*(tensor[1:] for tensor in tensors), 7, 2.0)
    spec_key = intj_mixed_bound.__self__.__self__.spec_key
    assert spec_key(*mixed_args) == spec_key(*other_args) == (b"", 5)

    cases = [
        ("FFI packed nop", packed_nop, converted),
        ("FFI typed nop", mod.typed_nop, converted),
        ("FFI empty kernel", mod.launch_empty, converted),
        ("INTJ empty kernel", intj_launch, (device, stream, grid, *tensors)),
        ("INTJ fixed-device kernel", intj_bound, (stream, grid, *tensors)),
        ("FFI packed nop mixed", packed_nop, mixed_converted),
        ("FFI typed nop mixed", mod.typed_nop_mixed, mixed_converted),
        ("FFI mixed kernel", mod.launch_mixed, mixed_converted),
        ("INTJ mixed kernel", intj_mixed, (device, stream, grid, *mixed_args)),
        ("INTJ fixed mixed kernel", intj_mixed_bound, (stream, grid, *mixed_args)),
    ]
    results: dict[str, list[float]] = {name: [] for name, *_ in cases}
    with tvm_ffi.use_raw_stream(converted[0].device, stream):
        for fn, args in (
            (mod.launch_mixed, mixed_converted),
            (intj_mixed, (device, stream, grid, *mixed_args)),
            (intj_mixed_bound, (stream, grid, *mixed_args)),
        ):
            fn(*args)
            torch.cuda.synchronize()
            assert tensors[2][0].item() == 6.5
        intj_mixed_bound(stream, grid, *other_args)
        torch.cuda.synchronize()
        assert tensors[2][1].item() == 8.0
        for sample in range(batches):
            for name, fn, args in reversed(cases) if sample % 2 else cases:
                results[name].append(bench(fn, args, iters, 1, torch.cuda.synchronize) / 1_000)

    print(f"iters={iters} batches={batches} unit=us/call (host-only, no timed sync)")
    for name, values in results.items():
        print(f"{name:<26} median={statistics.median(values):.4f} samples={values}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=10_000)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--sweep", action="store_true", help="compare zero to 64 arguments")
    parser.add_argument("--counts", type=int, nargs="+", default=[0, 3, 5, 8, 16, 32, 64])
    args = parser.parse_args()
    if args.iters < 1 or args.batches < 1:
        parser.error("--iters and --batches must be positive")
    if any(count < 0 for count in args.counts) or len(set(args.counts)) != len(args.counts):
        parser.error("--counts must be distinct nonnegative integers")
    if args.sweep:
        sweep(args.counts, args.iters, args.batches)
    else:
        main(args.iters, args.batches)
