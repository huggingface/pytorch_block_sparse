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
#include <cutlass/gemm/dispatch.h>
#include <cutlass/gemm/epilogue_function.h>

// Test utilities
#include "util/command_line.h"
#include "util/matrix.h"
#include "util/timer.h"
#include "util/type_conversion.h"
#include "util/bsc.h"

// Dispatch routines to CUTLASS
#include "cublas_dispatch.h"
#include "cutlass_dispatch.h"

using namespace std;
using namespace cutlass;


cublasHandle_t g_cublas_handle;
bool cublas_inited = false;

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
bool test_bsc(
    value_t* A_data,
    value_t* B_data,
    int* B_bsc_data,
    int* B_bsc_indices,
    accum_t* C_data,
    int m,          ///< Height of C in rows
    int n,          ///< Width of C in columns
    int k,          ///< Width (height) of A (B)
    accum_t alpha,  ///< Multiplicand scalar
    accum_t beta)
{
    cudaStream_t stream = 0;

    test_func_t test_func;

    cudaError_t error = test_func(
        g_cublas_handle,
        m,
        n,
        k,
        A_data,
        B_data,
        B_bsc_data,
        B_bsc_indices,
        C_data,
        alpha,
        beta,
        stream,
        false).result;

    return error;
}

 torch::Tensor  blocksparse_matmul_transpose_cutlass(torch::Tensor dense_a,
								      torch::Tensor row_ends_b,
								      torch::Tensor cols_b,
								      torch::Tensor data_b,
								      int size_rows_b,
								      int size_cols_b,
								      int block_size_rows_b,
								      int block_size_cols_b,
								      torch::Tensor dense_out)
{
    typedef float       value_t;
	typedef float       accum_t;
	const math_operation_class_t math_op = math_operation_class_t::scalar;
    static const matrix_transform_t::kind_t TransformA = matrix_transform_t::NonTranspose;
    static const matrix_transform_t::kind_t TransformB = matrix_transform_t::NonTranspose;

    auto sizes_a = dense_a.sizes().vec();
    auto sizes_out = dense_out.sizes().vec();

    value_t* A_data = (value_t*)dense_a.data_ptr();
    value_t* B_data = (value_t*)data_b.data_ptr();
    int* B_bsc_data = (int*)cols_b.data_ptr();
    int* B_bsc_indices = (int*)row_ends_b.data_ptr();
    value_t* C_data = (value_t*)dense_out.data_ptr();

    int m = sizes_a[1]; // A is transposed in pytorch, non_transposed in cutlass speak
    int n = size_rows_b;
    int k = size_cols_b;

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


	bool test_error = test_bsc<
	cutlass_gemm_dispatch<gemm::tiling_strategy::Custom, math_op, TransformA, TransformB, value_t, accum_t>,
	gemm::tiling_strategy::Custom,
	TransformA,
	TransformB,
	value_t,
	accum_t>(A_data,B_data,B_bsc_data, B_bsc_indices, C_data, m, n, k, accum_t(alpha), accum_t(beta));

    return dense_out;
}
