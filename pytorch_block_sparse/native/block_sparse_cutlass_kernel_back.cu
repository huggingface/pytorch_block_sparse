#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <string>
#include <fstream>
#include <sstream>

// CUBLAS GEMM API
#include <cublas_v2.h>

// Cutlass GEMM API
#include <cutlass/util/util.h>
#include <cutlass/gemm/dispatch_back.h>
#include <cutlass/gemm/epilogue_function.h>

// Dispatch routines to CUTLASS
#include "cutlass_dispatch_back.h"

using namespace std;
using namespace cutlass;


/**
 * Compute C = A.matmul(B), where A and B are dense, and only on the sparse support of C
 **/
template <typename                        func_t,    ///< Test function type
          gemm::tiling_strategy::kind_t   TilingStrategy,
          matrix_transform_t::kind_t      TransformA,     ///< Transformation op for matrix A
          matrix_transform_t::kind_t      TransformB,     ///< Transformation op for matrix B
          typename                        value_t,        ///< Multiplicand value type (matrices A and B)
          typename                        accum_t         ///< Accumulator value type (matrix C and scalars)
         >
cudaError_t back_full(value_t* A_data,
 					  value_t* B_data,
					  accum_t* C_data,
					  int2* C_blocks,
					  long C_blocks_length,
					  int m,          ///< Height of C in rows
					  int n,          ///< Width of C in columns
					  int k)
{

    typedef gemm::gemm_policy<value_t, accum_t, TransformA, TransformB, TilingStrategy> block_task_back_policy_t;

    cudaStream_t stream = 0;

    func_t func;

    cudaError_t error = func(m,
							 n,
							 k,
							 A_data,
							 B_data,
							 C_data,
							 C_blocks,
							 C_blocks_length,
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
cudaError_t back(value_t* A_data,
 				 value_t* B_data,
			     accum_t* C_data,
				 int2* C_blocks,
				 long C_blocks_length,
				 int m,          ///< Height of C in rows
				 int n,          ///< Width of C in columns
				 int k)

{
  cudaError_t error = back_full<cutlass_gemm_dispatch_back<gemm::tiling_strategy::CustomBack,
      			                                           math_operation_class_t::scalar,
			                                               TransformA,
									                       TransformB,
										                   value_t,
											               accum_t>,
  							    gemm::tiling_strategy::CustomBack,
							    TransformA,
							    TransformB,
							    value_t,
							    accum_t>(A_data, B_data, C_data, C_blocks, C_blocks_length, m, n, k);
  return error;
}

typedef cudaError_t (*back_t)(float* A_data,
							  float* B_data,
							  float* C_data,
							  int2* C_blocks,
							  long C_blocks_length,
							  int m,
							  int n,
							  int k);

/**
 * matrix a must be of dimensions [m,k]
 * matrix b must be of dimensions [k,n]
 * if pytorch_contiguous_a is true, then dense_a must be contiguous, ortherwise  dense_a.t() must be contiguous.
 * if pytorch_contiguous_b is true, then dense_b must be contiguous, ortherwise  dense_b.t() must be contiguous.
 **/
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
									long sparse_blocks_length_c)
{
    typedef float       value_t;
	typedef float       accum_t;

    value_t* A_data = (value_t*)dense_a.data_ptr();
    value_t* B_data = (value_t*)dense_b.data_ptr();
    value_t* C_data = (value_t*)sparse_c.data_ptr();
    int2* C_blocks = (int2*)sparse_blocks_c.data_ptr();
    long C_blocks_length = sparse_blocks_length_c;

    back_t back_fun;

    static const matrix_transform_t::kind_t NonTranspose = matrix_transform_t::NonTranspose;
    static const matrix_transform_t::kind_t Transpose = matrix_transform_t::Transpose;

    if (pytorch_contiguous_a) {
        if (pytorch_contiguous_b) {
             back_fun = back<Transpose, Transpose, value_t, accum_t>;
        } else {
             back_fun = back<Transpose, NonTranspose, value_t, accum_t>;
        }
    } else {
        if (pytorch_contiguous_b) {
            back_fun = back<NonTranspose, Transpose, value_t, accum_t>;
        } else {
            back_fun = back<NonTranspose, NonTranspose, value_t, accum_t>;
        }
    }

    return back_fun(A_data,B_data, C_data, C_blocks, C_blocks_length, m, n, k);
}
