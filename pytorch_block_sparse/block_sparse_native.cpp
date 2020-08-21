#include <torch/extension.h>
#include <vector>


torch::Tensor blocksparse_matmul_cutlass(torch::Tensor dense_a,
 												   torch::Tensor row_ends_b,
												   torch::Tensor cols_b,
												   torch::Tensor data_b,
												   int m,
												   int n,
								                   int k,
												   int block_size_rows_b,
												   int block_size_cols_b,
												   torch::Tensor dense_out);


torch::Tensor  blocksparse_matmul_back_cutlass(torch::Tensor dense_a,
											  torch::Tensor dense_b,
											  int m,
											  int n,
											  int k,
											  int block_size_rows_b,
											  int block_size_cols_b,
											  torch::Tensor sparse_c,
											  torch::Tensor sparse_c_blocks,
											  long sparse_c_blocks_length
											  );


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("blocksparse_matmul_cutlass", &blocksparse_matmul_cutlass, "blocksparse_matmul_cutlass");
  m.def("blocksparse_matmul_back_cutlass", &blocksparse_matmul_back_cutlass, "blocksparse_matmul_back_cutlass");

}