#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Inspired by https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/tensor-cores/simpleTensorCoreGEMM.cu


torch::Tensor blocksparse_matmul_cuda(torch::Tensor dense_a,
								      torch::Tensor col_ends_b,
								      torch::Tensor rows_b,
								      torch::Tensor data_b,
								      int size_rows_b,
								      int size_cols_b,
								      int block_size_rows_b,
								      int block_size_cols_b,
								      bool transpose_b)
{
    auto sizes_a = dense_a.sizes().vec();
    const dim3 blocks(size_cols_b / block_size_cols_b, sizes_a[0] / block_size_rows_b);

    printf("%d, %d", blocks.x, blocks.y);

    return dense_a;
}
