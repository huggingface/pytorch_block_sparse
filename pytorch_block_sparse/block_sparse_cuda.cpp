#include <torch/extension.h>

#include <vector>

// CUDA forward declarations


torch::Tensor blocksparse_matmul_cuda(torch::Tensor a_dense,
								      torch::Tensor b_cols,
								      torch::Tensor b_row_end,
								      torch::Tensor b_data,
								      int b_size_rows,
								      int b_size_cols,
								      int b_block_size_rows,
								      int b_block_size_cols);

torch::Tensor blocksparse_matmul(torch::Tensor a_dense,
                                 torch::Tensor b_cols,
                                 torch::Tensor b_row_end,
                                 torch::Tensor b_data,
                                 int b_size_rows,
								 int b_size_cols,
								 int b_block_size_rows,
								 int b_block_size_cols)
{
    return blocksparse_matmul_cuda(a_dense, b_cols, b_row_end, b_data, b_size_rows, b_size_cols, b_block_size_rows, b_block_size_cols);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("blocksparse_matmul", &blocksparse_matmul, "blocksparse_matmul");
}