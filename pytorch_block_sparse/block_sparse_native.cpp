#include <torch/extension.h>

#include <vector>

// CUDA forward declarations


torch::Tensor blocksparse_matmul_transpose_cuda(torch::Tensor dense_a,
								      torch::Tensor row_ends_b,
								      torch::Tensor cols_b,
								      torch::Tensor data_b,
								      int size_rows_b,
								      int size_cols_b,
								      int block_size_rows_b,
								      int block_size_cols_b,
								      torch::Tensor dense_out);


torch::Tensor blocksparse_matmul_transpose_cublas(torch::Tensor dense_a,
								      torch::Tensor dense_b,
								      torch::Tensor dense_out);

torch::Tensor blocksparse_matmul_transpose_sparse(torch::Tensor dense_a,
                                 torch::Tensor row_ends_b,
                                 torch::Tensor cols_b,
                                 torch::Tensor data_b,
                                 int size_rows_b,
								 int size_cols_b,
								 int block_size_rows_b,
								 int block_size_cols_b,
								 torch::Tensor dense_out)
{
    return blocksparse_matmul_transpose_cuda(dense_a, row_ends_b, cols_b, data_b, size_rows_b, size_cols_b, block_size_rows_b, block_size_cols_b, dense_out);
}

torch::Tensor blocksparse_matmul_transpose_dense(torch::Tensor dense_a,
                                                 torch::Tensor dense_b,
   								                 torch::Tensor dense_out)
{
    return blocksparse_matmul_transpose_cublas(dense_a, dense_b, dense_out);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("blocksparse_matmul_transpose_dense", &blocksparse_matmul_transpose_dense, "blocksparse_matmul_transpose_dense");
  m.def("blocksparse_matmul_transpose_sparse", &blocksparse_matmul_transpose_sparse, "blocksparse_matmul_transpose_sparse");
}