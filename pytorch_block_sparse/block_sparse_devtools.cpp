#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "util/bsc.h"

using namespace std;
using namespace cutlass;

// dense_b is a block sparse matrix, on CPU, with float type
// we translate it to the original cutlass_sparse bsc object and returns the index and data vectors
// so we can check that we are building compatible ones, but with pytorch code
std::vector<torch::Tensor> bsc_build_from_dense(int NBlocks,
                                                int KBlocks,
                                                int nonZeroBlocks,
                                                int BlockItemsN,
                                                int BlockItemsK,
                                                torch::Tensor dense_b,
												int k,
												int n)
{
   auto ret = std::vector<torch::Tensor>();
   typedef float scalar_t;

   scalar_t* dense_b_data = (scalar_t*)dense_b.data_ptr();

   bsc<scalar_t> B_bsc(NBlocks, KBlocks, nonZeroBlocks, BlockItemsN, BlockItemsK, dense_b_data, k, n);

   scalar_t* data = B_bsc.h_data();
   long data_size = B_bsc.h_data_size();

   torch::Tensor data_tensor = torch::from_blob(data, data_size).clone();

   auto intOptions = torch::TensorOptions().dtype(torch::kInt32);

   int* ptr = B_bsc.h_ptr();
   long ptr_size = B_bsc.h_ptr_size();

   torch::Tensor ptr_tensor = torch::from_blob(ptr, ptr_size, intOptions).clone();

   int* indices = B_bsc.h_indices();
   long indices_size = B_bsc.h_indices_size();

   torch::Tensor indices_tensor = torch::from_blob(indices, indices_size, intOptions).clone();

   ret.push_back(data_tensor);
   ret.push_back(ptr_tensor);
   ret.push_back(indices_tensor);

   return ret;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bsc_build_from_dense", &bsc_build_from_dense, "bsc_build_from_dense");

}