#import "@preview/fireside:1.0.0": fireside

#show: fireside.with(
  title: [Little Booklet on Writing PyTorch CUDA extensions],
  from-details: [
    Da1sypetals \
    Da1sypetals.iota\@gmail.com \
  ],
  to-details: [
    #v(-8%)
  ],
)

#set heading(numbering: "1.")
#set text(top-edge: 0.7em, bottom-edge: -0.3em)
#show link: set text(fill: blue)

// The below topics are experiences from production. Read this before you write your own PyTorch CUDA extension, whether or not it will be used in production. I am sure this will save you at least 3 days' debugging.

= Check tensor storage

== Device check
You should ALWAYS check EXPLICITLY whether input tensors are on desired devices. In most cases you want them on *the same GPU*, or in rare cases you want some tensors on CPU to perform some operations that are not efficient on GPU.

API:
- `tensor.is_cuda()`
- `tensor.device()`
  - use `operator=` for equality comparison.

Sometimes the _not on correct device_ problem causes strange error messages like `Cusparse context initialization failure` or things even more weird, which first seem unrelated to a device problem. This is why I suggest you always start your debug journey here.

== Contiguity check
Modern LibTorch recommends using #link("https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/TensorAccessor.h", "Packed tensor accessor") (roughly the same memory cost as a pointer) to access elements in tensor.

However, If you are to plug some others' implementation (likely using raw pointers like `float*`) into PyTorch, you are not likely to understand the code inside out and rewrite it.

Usually, in the context of deep learning, most implementations assumes a *row-major contiguous* storage. You should explicitly check whether the input tensors are contiguous in the C++ code that wraps the CUDA kernel.

API: `tensor.is_contiguous()`

== Cheatsheet
A quick utility that checks whether all tensors are on *the same CUDA device*:
```cpp
void CheckInputTensors(const std::vector <torch::Tensor> &tensors) {
    TORCH_CHECK(!tensors.empty(), "No tensors provided for device check");

    auto first_device = tensors[0].device();
    TORCH_CHECK(first_device.is_cuda(), "First tensor is not on CUDA");

    int idx = 0;
    for (const auto &tensor: tensors) {
        TORCH_CHECK(tensor.device() == first_device,
                    "All tensors must be on the same CUDA device, "
                    "but found tensor at index [", idx,
                    "] on device ", tensor.device(),
                    " while expecting ", first_device);

        TORCH_CHECK(tensor.is_contiguous(),
                    "All tensors must be contiguous, but found tensor at index [",
                    idx, "] not contiguous");

        idx += 1;
    }
}
```

= CUDA toolkit version problem
Most "symbol not found" problem are caused by compiler / assembler / library version mismatch. Let me elaborate on this a bit:

+ PyTorch has an important version information attached to it: *_The version of CUDA that torch is compiled on (let's call it CVT, Cuda Version of Torch_, for the sake of simplicity)*. The torch installation comes with its own CUDA toolkit (that matches CVCT) with *no nvcc, ptxas*.
+ If you are to write custom CUDA extension to PyTorch, *it will use the nvcc and ptxas in your system `PATH`, and libraries like CUBLAS or CUSPARSE in `LD_LIBRARY_PATH`*. Let's call this CUDA toolkit version *_CVE, Cuda Version of Extension_*.

+ When you try to compile a CUDA extension, #text(size: 14pt, weight: "bold")[Make sure that your CVT and CVE *perfectly match* (NOT major version match).]
  - When you compile your extension, PyTorch hints you that a minor version mismatch should not be a problem. *Remember, everything should not happen will eventually happen.*

= Debug layer by layer

A CUDA extension is roughly split into 4 parts, from the bottom to the top namely:
- CUDA kernel
- C++ wrapper
- data passed from Python (PyTorch) to C++
- Python wrapper

== CUDA kernel
Debugging CUDA kernel is a very very difficult problem and we shall not discuss it here.

== C++ wrapper
The first thing I want to hint you is that do not dereference a pointer pointing to device in host functions. You should always mark device pointers with a `d_` prefix in variable names, or wrap it with `thrust::device_ptr`.

`printf`, `std::cout` or `gdb` will assist you in the journey.

== data passed from Python (PyTorch) to C++
Refer to Pybind11 docs and try to answer these questions:

- How various Python types are represented in Pybind11 API;
- How to properly configure the function prototype in Pybind11?


== Python Wrapper
Ask LLMs. LLMs know python much better than I do.


= Using CUBLAS, CUSPARSE, CUSolverDn, _etc_.


#text(
  fill: color.linear-rgb(33, 33, 33, 255),
)[_We use CUSPARSE as an example. The same rule apply to other libraries like CUBLAS or CUSolverDn._]

== Handles

When writing pure CUDA/C++ code, you manually call `cusparseCreate` to initialize the CUSPARSE context and prepare for subsequent CUSPARSE API calls.

However this is not best practice in PyTorch CUDA extensions. There are good reasons: `cusparseCreate` introduces a milliseconds-level delay on CPU side. This may not be noticeable at first, but remember that operators are written to be run millions of times, which turns this into a significant overhead. This can cause GPU to starve when waiting CPU for synchronization.

- If you use `VizTracer` to trace your program and visualize it in #link("ui.perfetto.dev", "perfetto"), you may notice `cudaGetDeviceProperties` call taking too much time on CPU side. This can be directly caused by `cusparseCreate`.

LibTorch has API that automatically manages a pool of CUSPARSE handles:

+ Include the header that brings in CUDA context manager for LibTorch:
  ```cpp
  #include <ATen/cuda/CUDAContext.h>
  ```

+ Then, get handle via
  ```cpp
  auto handle = at::cuda::getCurrentCUDASparseHandle();
  ```
  `getCurrentCUDASparseHandle` automatically create a handle if there is not any, amc caches it for subsequent uses.

+ Use your handle as usual.

I could not find documentation for these APIs, so if you want to know more, you may need to read the source code of PyTorch `ATen`. Searching in the repo with keyword `getcurrentcuda` can get you there quickly.

#image("handles.png", width: 80%)



= Tensor Options

`struct TensorOptions` carries many information about the tensor:

```cpp
struct C10_API TensorOptions {

  // ... omitted

  // members
  Device device_ = at::kCPU; // 16-bit
  caffe2::TypeMeta dtype_ = caffe2::TypeMeta::Make<float>(); // 16-bit
  Layout layout_ = at::kStrided; // 8-bit
  MemoryFormat memory_format_ = MemoryFormat::Contiguous; // 8-bit

  bool requires_grad_ : 1;
  bool pinned_memory_ : 1;

  // Existense of members
  bool has_device_ : 1;
  bool has_dtype_ : 1;
  bool has_layout_ : 1;
  bool has_requires_grad_ : 1;
  bool has_pinned_memory_ : 1;
  bool has_memory_format_ : 1;
}
```

The most important methods are

```cpp
[[nodiscard]] TensorOptions device(Device device) const;
[[nodiscard]] TensorOptions dtype(ScalarType dtype) const;
[[nodiscard]] TensorOptions requires_grad(bool) const;
```

Usage:
- `tensor.options()` returns an instance of `TensorOptions` that describes the `tensor`.
- `opt.dtype(torch::kFloat64)` has other properties remain the same as `opt`, only `dtype` changes to `float64` or in C++, `double`.
- The `.to(...)` method of a tensor can take a `TensorOptions` instance as its only argument.


For an exhaustive list of device and dtype, you may want to refer to:
- https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/types.h

- https://github.com/pytorch/pytorch/blob/main/c10/core/DeviceType.h




= What to Reference

To my knowledge, the PyTorch C++ #link("https://pytorch.org/cppdocs/api/library_root.html", "documentation") is very old. Many things in the source code are not documented there.

It is a better choice to just search in the PyTorch #link("https://github.com/pytorch/pytorch", "github repo"), and read the comments and source code.
