#include <torch/extension.h>
#include <vector>
#include <cuda.h>


int blocksparse_matmul_cutlass(torch::Tensor dense_a,
                               bool pytorch_contiguous_a,
  							   torch::Tensor ptr_b,
							   torch::Tensor indices_b,
							   torch::Tensor data_b,
							   int m,
							   int n,
							   int k,
							   int block_size_rows_b,
							   int block_size_cols_b,
							   torch::Tensor dense_out);

int blocksparse_matmul_back_cutlass(torch::Tensor dense_a,
                                    bool pytorch_contiguous_a,
									torch::Tensor dense_b,
                                    bool pytorch_contiguous_b,
									int m,
									int n,
									int k,
									int block_size_rows_b,
									int block_size_cols_b,
									torch::Tensor sparse_c,
									torch::Tensor sparse_blocks_c,
									long sparse_blocks_length_c);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("blocksparse_matmul_cutlass", &blocksparse_matmul_cutlass, "blocksparse_matmul_cutlass");
  m.def("blocksparse_matmul_back_cutlass", &blocksparse_matmul_back_cutlass, "blocksparse_matmul_back_cutlass");
}