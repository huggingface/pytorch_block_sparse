#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <THC/THCAtomics.cuh>

// Inspired by https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/tensor-cores/simpleTensorCoreGEMM.cu
template <typename scalar_t>
__global__ void blocksparse_matmul_transpose_kernel_ref(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dense_a,
                                                    const int64_t rows_a,
                                                    const int64_t cols_a,
                                                    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_ends_b,
                                                    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> cols_b,
                                                    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> data_b,
                                                    const int block_size_rows_b,
                                                    const int block_size_cols_b,
                                                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dense_out) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;

  int row_start = row_ends_b[blockIdx.y];
  int row_end = row_ends_b[blockIdx.y + 1];

  scalar_t accumulator = 0.0;

  for(int i = row_start; i < row_end ; i ++) {
      int column = cols_b[i * 2];
      int block_offset = cols_b[i * 2 + 1];

      for(int j = 0; j < block_size_cols_b; j++) {
          accumulator += data_b[block_offset * block_size_rows_b + threadIdx.y][j] * dense_a[r][column * block_size_cols_b + j];
      }
  }

  dense_out[r][c] = accumulator;
}

// Inspired by https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/tensor-cores/simpleTensorCoreGEMM.cu
template <typename scalar_t>
__global__ void blocksparse_matmul_transpose_kernel(const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dense_a,
                                                    const int64_t rows_a,
                                                    const int64_t cols_a,
                                                    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_ends_b,
                                                    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> cols_b,
                                                    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> data_b,
                                                    const int block_size_rows_b,
                                                    const int block_size_cols_b,
                                                    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dense_out) {
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;

  const int row_start = row_ends_b[blockIdx.x];
  const int row_end = row_ends_b[blockIdx.x + 1];
  __shared__ scalar_t fshare_a[16][16];
  __shared__ scalar_t fshare_b[16][16];

  scalar_t accumulator = 0.0;

  #pragma unroll
  for(int i = row_start; i < row_end ; i ++) {
      const int block_offset_row = cols_b[i * 2 + 1] * block_size_rows_b + threadIdx.y; // Change to threadIdx.x
      const int block_offset_col = cols_b[i * 2] * block_size_cols_b;

      __syncthreads();

      fshare_a[threadIdx.y][threadIdx.x] = dense_a[r][block_offset_col + threadIdx.x];
      fshare_b[threadIdx.y][threadIdx.x] = data_b[block_offset_row][threadIdx.x];

      __syncthreads();

      scalar_t* ptr_a = fshare_a[threadIdx.y];
      scalar_t* ptr_b = fshare_b[threadIdx.x];

      #pragma unroll

      for(int j = 0; j < block_size_cols_b; j++) {
          accumulator += ptr_a[j] * ptr_b[j];
      }

  }

  dense_out[r][c] = accumulator;
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

    //printf("%d, %d\n", sizes_a[0], sizes_a[1]);

    const dim3 blocks(sizes_out[1] / block_size_cols_b,
                      sizes_out[0] / block_size_rows_b);
    const dim3 threads_per_block(block_size_cols_b, block_size_rows_b);

    //printf("blocks %d %d\n", blocks.x, blocks.y);
    //printf("threads_per_block %d %d\n", threads_per_block.x, threads_per_block.y);

     AT_DISPATCH_FLOATING_TYPES(dense_out.scalar_type(), "blocksparse_matmul_cuda", ([&] {
       blocksparse_matmul_transpose_kernel<scalar_t><<<blocks, threads_per_block>>>(
           dense_a.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
           sizes_a[0], sizes_a[1],
           row_ends_b.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
           cols_b.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
           data_b.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
           block_size_rows_b, block_size_cols_b,
           dense_out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()
           );
            }));


    return dense_out;
}
