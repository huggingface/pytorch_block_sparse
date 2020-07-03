#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <THC/THCAtomics.cuh>


// Inspired by https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/tensor-cores/simpleTensorCoreGEMM.cu
template <typename scalar_t>
__global__ void blocksparse_matmul_transpose_kernel(torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dense_a,
                             torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_ends_b,
                             torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> cols_b,
                             torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> data_b,
                             torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dense_out) {
/*
  int oColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int oRow    = blockIdx.y * blockDim.y + threadIdx.y;
  int oFrame  = (blockIdx.z + offsetZ) % output.size(1); // output frame/time
  int slice   = (blockIdx.z + offsetZ) / output.size(1); // output slice/feature
*/
 //const int n = blockIdx.y;
  // column index
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  //int i = threadIdx.x;
  dense_out[x][y] *= 2.0f;
}

torch::Tensor blocksparse_matmul_transpose_cuda(torch::Tensor dense_a,
								      torch::Tensor row_ends_b,
								      torch::Tensor cols_b,
								      torch::Tensor data_b,
								      int size_rows_b,
								      int size_cols_b,
								      int block_size_rows_b,
								      int block_size_cols_b,
								      torch::Tensor dense_out)
{
    auto sizes_out = dense_out.sizes().vec();
    auto sizes_a = dense_a.sizes().vec();
    //const dim3 blocks(size_cols_b / block_size_cols_b, sizes_a[0] / block_size_rows_b);

    //printf("%d, %d", blocks.x, blocks.y);

    const int threads = 32;
    const dim3 blocks(sizes_out[0] / 16,
                      sizes_out[1] / block_size_rows_b,
                      1);
    const dim3 threads_per_block(threads, threads, 1);

     AT_DISPATCH_FLOATING_TYPES(dense_out.scalar_type(), "blocksparse_matmul_cuda", ([&] {
       blocksparse_matmul_transpose_kernel<scalar_t><<<blocks, threads_per_block>>>(
           dense_a.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
           row_ends_b.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
           cols_b.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
           data_b.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
           dense_out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
            }));


    return dense_out;
}
