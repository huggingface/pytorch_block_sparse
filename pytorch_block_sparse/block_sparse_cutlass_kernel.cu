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
 * Compute C = (alpha * A * B) + (beta * C), where B is block sparse, A and B dense
 **/
template <
    typename                        func_t,    ///< Test function type
    gemm::tiling_strategy::kind_t   TilingStrategy,
    matrix_transform_t::kind_t      TransformA,     ///< Transformation op for matrix A
    matrix_transform_t::kind_t      TransformB,     ///< Transformation op for matrix B
    typename                        value_t,        ///< Multiplicand value type (matrices A and B)
    typename                        accum_t>        ///< Accumulator value type (matrix C and scalars)
cudaError_t forward(
    value_t* A_data,
    value_t* B_data,
    int* B_bsc_ptr,
    int* B_bsc_indices,
    accum_t* C_data,
    int m,          ///< Height of C in rows
    int n,          ///< Width of C in columns
    int k,          ///< Width (height) of A (B)
    accum_t alpha,  ///< Multiplicand scalar
    accum_t beta)
{

    typedef gemm::gemm_policy<value_t, accum_t, TransformA, TransformB, TilingStrategy> block_task_policy_t;

    // matrix pruning
    int BlockItemsN = block_task_policy_t::BlockItemsX; // depend on the block task policy
    int BlockItemsK = block_task_policy_t::BlockItemsK;

    cudaStream_t stream = 0;

    func_t func;

    cudaError_t error = func(
        m,
        n,
        k,
        A_data,
        B_data,
        B_bsc_ptr,
        B_bsc_indices,
        C_data,
        alpha,
        beta,
        stream,
        false).result;

    return error;
}

int blocksparse_matmul_cutlass(torch::Tensor dense_a,
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
	const math_operation_class_t math_op = math_operation_class_t::scalar;
    static const matrix_transform_t::kind_t TransformA = matrix_transform_t::Transpose;
    static const matrix_transform_t::kind_t TransformB = matrix_transform_t::Transpose;

    value_t* A_data = (value_t*)dense_a.data_ptr();
    value_t* B_data = (value_t*)data_b.data_ptr();
    int* B_ptr = (int*)ptr_b.data_ptr();
    int* B_indices = (int*)indices_b.data_ptr();
    value_t* C_data = (value_t*)dense_out.data_ptr();

    float alpha = 1.0;
    float beta = 0.0;

	cudaError_t error = forward<cutlass_gemm_dispatch<gemm::tiling_strategy::Custom,
	                                                 math_op,
	                                                 TransformA,
	                                                 TransformB,
	                                                 value_t,
	                                                 accum_t>,
						       gemm::tiling_strategy::Custom,
					           TransformA,
						       TransformB,
						       value_t,
						       accum_t>(A_data,B_data,B_ptr, B_indices, C_data, m, n, k, accum_t(alpha), accum_t(beta));
    return error;
}


