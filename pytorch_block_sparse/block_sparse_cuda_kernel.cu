#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Inspired by https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/tensor-cores/simpleTensorCoreGEMM.cu


torch::Tensor blocksparse_matmul_cuda(torch::Tensor a_dense,
								      torch::Tensor b_cols,
								      torch::Tensor b_row_end,
								      torch::Tensor b_data,
								      int b_size_rows,
								      int b_size_cols,
								      int b_block_size_rows,
								      int b_block_size_cols)
{
    auto a_sizes = a_dense.sizes().vec();
    const dim3 blocks(b_size_cols / b_block_size_cols, a_sizes[0] / b_block_size_rows);

    printf("%d, %d", blocks.x, blocks.y);

    return a_dense;
}
