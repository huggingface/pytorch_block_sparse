# Development Notes


This python package provides a PyTorch extension .


## Organisation
### Build

The setup.py script use the standard PyTorch extension mechanism to build the package:

```
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
...
      ext_modules=[
        CUDAExtension('block_sparse_native',
                      ['pytorch_block_sparse/native/block_sparse_native.cpp',
                      'pytorch_block_sparse/native/block_sparse_cutlass_kernel_back.cu',
                      'pytorch_block_sparse/native/block_sparse_cutlass_kernel.cu'],
                      extra_compile_args=['-I', '%s/pytorch_block_sparse' % rootdir]
                      ),
      ],
      cmdclass={
        'build_ext': BuildExtension
      }
```

### Native functions python interface
A single c++ file `block_sparse_native.cpp` provides the native functions visible from python.
These functions provides access to CUDA kernels which computes :
 - dense x native -> dense
 - dense x dense on sparse support -> sparse

### CUDA/Cutlass kernels
The `*.cu` files in the `native` directory provides the kernel themselves.
They are using the cutlass primitives available in the `cutlass` subdirectory.

Multiple levels of C++ templating provides dispatch/code generation of the kernels.

The main files in the `cutlass/gemm` directory are `block_task.h` and `block_task_back.h` .
They express the final CUDA kernel that will be executed, using 
- `block_loader_.*` to load A and B matrix tiles in an efficient way
- `thread_accumulator.h` to store the result tiles 'R'
- `epilogue_function` to combine R with C  `C' = alpha * R + beta * C`
- `grid_raster_.*` to list the output tiles that must be computed

### block_sparse python module
This library includes as little native code as possible, because native code is hard to write/debug/understand.

The native functions are performing the performance critical tasks, and the python code in `block_sparse.py` is doing
all the preparatory work, which is executed only once, or a unfrequently.

The main job of `block_sparse.py` is to build indexes into the sparse matrices.
Three sets of sparse indices are built:
- row wise index of non-zero entries (for dense x sparse)
- column wise index of non-zero entries (for dense x sparse with transposition)
- linear list of 2D coordinates of non-zero entries (for dense x dense on sparse support)

These structures are created using standard PyTorch primitives, and so are easy to debug, understand,
or reimplement in other languages.

### block_sparse_linear python module
The block_sparse_linear is a thin layer on top of `block_sparse`
It use the linear algebra primitives of block_sparse to create a drop in replacement for `torch.nn.Linear`,
with the proper back-propagation primitives, implemented using a `torch.autograd.Function` subclass. 

## Testing 
Debugging CUDA kernels is hard. Fortunately, it's easy to compare the kernel results with
a reference PyTorch implementation.
The `tests` directory provides some code to test and measure performance of the library.

## TODO

block_sparse
- add input parameters sanity checks
- add dispatch for 
  - different matrix size -> different dispatch strategy (tile sizes in k-dimension)
  - different block sizes
  
tests  
  - Refactor/cleanup tests
    
doc
- schema of sparse index structures

cutlass
- move to 2.x version

cleanup algorithms
- add algorithms to measure weights importance and optimize the sparsity pattern





