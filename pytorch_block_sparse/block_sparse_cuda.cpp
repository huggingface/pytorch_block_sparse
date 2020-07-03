#include <torch/extension.h>

#include <vector>

// CUDA forward declarations


torch::Tensor blocksparse_matmul_cuda(torch::Tensor dense_a,
								      torch::Tensor col_ends_b,
								      torch::Tensor rows_b,
								      torch::Tensor data_b,
								      int size_rows_b,
								      int size_cols_b,
								      int block_size_rows_b,
								      int block_size_cols_b,
								      bool tranpose_b);

torch::Tensor blocksparse_matmul(torch::Tensor dense_a,
                                 torch::Tensor col_ends_b,
                                 torch::Tensor b_row_end,
                                 torch::Tensor data_b,
                                 int size_rows_b,
								 int size_cols_b,
								 int block_size_rows_b,
								 int block_size_cols_b,
								 bool transpose_b)
{
    return blocksparse_matmul_cuda(dense_a, col_ends_b, b_row_end, data_b, size_rows_b, size_cols_b, block_size_rows_b, block_size_cols_b, transpose_b);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("blocksparse_matmul", &blocksparse_matmul, "blocksparse_matmul");
}