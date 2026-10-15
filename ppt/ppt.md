# Optimizing Triton Kernel Launch Overhead

## Motivation

## Method

### Usage example

An usage example, from aiter migration, show the diff and how the decorator is used.

## Results

### Performance

#### Benchmark results

##### Varing Argument/Constexpr-Argument/Cached-Kernel Number
##### Citing TVM FFI and Tilelang
cite https://tilelang.com/blog/2025/12/11/ffi.html, highlight date of the work
show the tilang part of figure 2
cite from the blog:

Every time the user accesses torch.Tensor.shape (or device, stride, …) in Python, it adds around 0.1–1 µs of overhead, which is painful when kernels are very fast.

after the SplitHostDevice pass, TileLang splits the IR into device and host parts, ..., finally, via codegen, we generate the C++ host code.

 For a DSL, CPU overhead is mostly kernel host and cache; TileLang leaves very little headroom in these two parts. (In contrast, due to design choices, Triton is weaker on these two aspects. In our experience Triton compiles faster, but its runtime CPU overhead is higher. We won’t benchmark here; please feel free to try.)

cite figure 3, figure 4

##### Comparing with TVM FFI
##### AITER KDA/fp8_mqa_logits

### AITER adoption

## Implementation Details

### Incompatible

#### Launch Syntax

- Gain:
- Price:

micro benchmark results for overhead of:

1.
```
triton.cdiv(intj compile it to C++)
```
2.
```
def __getitem__(self, grid) -> T:
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
```

#### ptr_range

triton:
```
if hasattr(arg, "ptr_range"):  return arg.ptr_range() <= 2**31-1     # duck-typed objects
if isinstance(arg, torch.Tensor): return arg.untyped_storage().size() <= 2**31-1
```
intj: get tensor range using pytorch C++.

showing micro benchmark results of triton `ptr_range()` overhead.

### Lazy Building

- Gain:
- Price:

### Triton Autotune/Heuristics Support

the necessity of nested hash table.

### Dependency on CPython/PyTorch/Triton internal

#### Test Matrix

- OS/CPython/PyTorch combination
- CPU Mocking, fake driver

### Micro Optimization

####

- CPython VectorCall and specialization; Use a quick benchmark loop.
- Quick hashtable at sparse load. intj/tsl/absl, low load factor microbenchmark results

#### TVM FFI Per Argument HashTable Lookup

per-argument, tvm ffi has a std::unordered_map lookup overhead.

- `PyObject* -> TVMFFIAny*`: lookup PyTypeObject* from std::unordered_map, `TVMFFIPySetArgumentGenericDispatcher/SetArgument`.
in python/tvm_ffi/cython/tvm_ffi_python_helpers.h
```
PyTypeObject* py_type = Py_TYPE(py_arg);
if (TVM_FFI_PREDICT_TRUE(it != arg_dispatch_map_.end())) {
    TVMFFIPyArgSetter setter = it->second;
    // if error happens, propagate it back
    if (TVM_FFI_PREDICT_FALSE(setter(ctx, py_arg, out) != 0)) return -1;
} ...
```
- `TVMFFIAny -> DLPackTensor/int/etc`: type-driven cast, zero overhead, `TypeTraits<T>::TryCastFromAnyView`.
in include/tvm/ffi/any.h
```
std::optional<T> opt = TypeTraits<T>::TryCastFromAnyView(&data_);
```

show microbenchmark results for the overhead (both light loaded and heavy loaded unordered_map)


### Annotation

## Discussion

Next Step:
1. merge the aiter pr
2. better doc, source code (10K agent generated code)
