#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
#include <iostream>
#include <typeinfo>
#include <stdint.h>
#include <string>
#include <fstream>
#include <sstream>
*/

// CUBLAS GEMM API
#include <cublas_v2.h>

// Cutlass GEMM API
#include <cutlass/util/util.h>
#include <cutlass/gemm/dispatch.h>
#include <cutlass/gemm/epilogue_function.h>

// Dispatch routines to CUTLASS
#include "cutlass_dispatch.h"

using namespace std;
using namespace cutlass;

/**
 * Compute C = A * B, where B is block sparse, A and C dense
 **/
template <
    typename                        func_t,    ///< Test function type
    gemm::tiling_strategy::kind_t   TilingStrategy,
    matrix_transform_t::kind_t      TransformA,     ///< Transformation op for matrix A
    matrix_transform_t::kind_t      TransformB,     ///< Transformation op for matrix B
    typename                        value_t,        ///< Multiplicand value type (matrices A and B)
    typename                        accum_t>        ///< Accumulator value type (matrix C and scalars)
cudaError_t forward_full(value_t* A_data,
                         value_t* B_data,
						 int* B_ptr,
						 int* B_indices,
						 accum_t* C_data,
						 int m,          ///< Height of C in rows
						 int n,          ///< Width of C in columns
						 int k          ///< Width (height) of A (B)
                         )
{
    typedef gemm::gemm_policy<value_t, accum_t, TransformA, TransformB, TilingStrategy> block_task_policy_t;

    cudaStream_t stream = 0;

    func_t func;

    cudaError_t error = func(m,
							 n,
							 k,
							 A_data,
							 B_data,
							 B_ptr,
							 B_indices,
							 C_data,
							 accum_t(1.0),
							 accum_t(0.0),
							 stream,
							 false).result;

    return error;
}

/**
 * Compute C = A.matmul(B), where A and B are dense, and only on the sparse support of C
 **/
template <matrix_transform_t::kind_t      TransformA,     ///< Transformation op for matrix A
          matrix_transform_t::kind_t      TransformB,     ///< Transformation op for matrix B
          typename                        value_t,        ///< Multiplicand value type (matrices A and B)
          typename                        accum_t         ///< Accumulator value type (matrix C and scalars)
         >
cudaError_t forward(value_t* A_data,
					value_t* B_data,
					int* B_ptr,
					int* B_indices,
					accum_t* C_data,
					int m,
					int n,
					int k)

{
  const math_operation_class_t math_op = math_operation_class_t::scalar;

  cudaError_t error = forward_full<cutlass_gemm_dispatch<gemm::tiling_strategy::Custom,
	                                                 math_op,
	                                                 TransformA,
	                                                 TransformB,
	                                                 value_t,
	                                                 accum_t>,
								   gemm::tiling_strategy::Custom,
								   TransformA,
								   TransformB,
								   value_t,
								   accum_t>(A_data,B_data,B_ptr, B_indices, C_data, m, n, k);
  return error;
}

typedef cudaError_t (*forward_t)(float* A_data,
								float* B_data,
								int* B_ptr,
								int* B_indices,
								float* C_data,
								int m,
								int n,
								int k);

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
							   torch::Tensor dense_out)
{
    typedef float       value_t;
	typedef float       accum_t;
    //static const matrix_transform_t::kind_t TransformA = matrix_transform_t::Transpose;
    static const matrix_transform_t::kind_t TransformB = matrix_transform_t::Transpose;

    value_t* A_data = (value_t*)dense_a.data_ptr();
    value_t* B_data = (value_t*)data_b.data_ptr();
    int* B_ptr = (int*)ptr_b.data_ptr();
    int* B_indices = (int*)indices_b.data_ptr();
    value_t* C_data = (value_t*)dense_out.data_ptr();

    static const matrix_transform_t::kind_t NonTranspose = matrix_transform_t::NonTranspose;
    static const matrix_transform_t::kind_t Transpose = matrix_transform_t::Transpose;

    forward_t forward_fun;

    assert(pytorch_contiguous_a);
	//if (pytorch_contiguous_a) {
    //      forward_fun = forward<NonTranspose, TransformB, value_t, accum_t>;
    //} else {
        forward_fun = forward<Transpose, TransformB, value_t, accum_t>;
    //}
    cudaError_t error = forward_fun(A_data,B_data,B_ptr, B_indices, C_data, m, n, k);

    return error;
}


