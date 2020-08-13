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

// Set Cutlass debug macro to enable console printing of library errors
#define DEBUG

// Cutlass GEMM API
#include <cutlass/util/util.h>
#include <cutlass/gemm/dispatch_back.h>
#include <cutlass/gemm/epilogue_function.h>

// Test utilities
#include "util/command_line.h"
#include "util/matrix.h"
#include "util/timer.h"
#include "util/type_conversion.h"


// Dispatch routines to CUTLASS
#include "cutlass_dispatch_back.h"

using namespace std;
using namespace cutlass;


extern cublasHandle_t g_cublas_handle;
extern bool cublas_inited;

/**
 * Compute C = (alpha * A * B) + (beta * C)
 */
template <
    typename                        test_func_t,    ///< Test function type
    gemm::tiling_strategy::kind_t   TilingStrategy,
    matrix_transform_t::kind_t      TransformA,     ///< Transformation op for matrix A
    matrix_transform_t::kind_t      TransformB,     ///< Transformation op for matrix B
    typename                        value_t,        ///< Multiplicand value type (matrices A and B)
    typename                        accum_t>        ///< Accumulator value type (matrix C and scalars)
bool test_bsc_back(
    value_t* A_data,
    value_t* B_data,
    accum_t* C_data,
    int2* C_blocks,
    long C_blocks_length,
    int m,          ///< Height of C in rows
    int n,          ///< Width of C in columns
    int k,          ///< Width (height) of A (B)
    accum_t alpha,  ///< Multiplicand scalar
    accum_t beta)
{

    typedef gemm::gemm_policy<value_t, accum_t, TransformA, TransformB, TilingStrategy> block_task_back_policy_t;

    // matrix pruning
    int BlockItemsN = block_task_back_policy_t::BlockItemsX; // depend on the block task policy
    int BlockItemsK = block_task_back_policy_t::BlockItemsK;

    cudaStream_t stream = 0;

    test_func_t test_func;

    cudaError_t error = test_func(
        g_cublas_handle,
        m,
        n,
        k,
        A_data,
        B_data,
        C_data,
        C_blocks,
        C_blocks_length,
        alpha,
        beta,
        stream,
        false).result;

    return error;
}

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
											  )
{
    typedef float       value_t;
	typedef float       accum_t;
	const math_operation_class_t math_op = math_operation_class_t::scalar;
    static const matrix_transform_t::kind_t TransformA = matrix_transform_t::Transpose;
    static const matrix_transform_t::kind_t TransformB = matrix_transform_t::Transpose;

    value_t* A_data = (value_t*)dense_a.data_ptr();
    value_t* B_data = (value_t*)dense_b.data_ptr();
    value_t* C_data = (value_t*)sparse_c.data_ptr();
    int2* C_blocks = (int2*)sparse_c_blocks.data_ptr();
    long C_blocks_length = sparse_c_blocks_length;

    //int m = sizes_a[0];

    float alpha = 1.0;
    float beta = 0.0;

// Initialize cuBLAS
	if (!cublas_inited) {
		if (cublasCreate(&g_cublas_handle) != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "cublasCreate() failed\n");
			exit(1);
		}
		cublas_inited = true;
	}

	bool test_error = test_bsc_back<
	cutlass_gemm_dispatch_back<gemm::tiling_strategy::CustomBack, math_op, TransformA, TransformB, value_t, accum_t>,
	gemm::tiling_strategy::CustomBack,
	TransformA,
	TransformB,
	value_t,
	accum_t>(A_data,B_data, C_data, C_blocks, C_blocks_length, m, n, k, accum_t(alpha), accum_t(beta));

    return sparse_c;
}
