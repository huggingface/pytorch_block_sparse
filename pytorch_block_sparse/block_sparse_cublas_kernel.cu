#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuda.h>


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

// Dispatch routines to CUBLAS
#include "cublas_dispatch.h"

using namespace std;
using namespace cutlass;



/******************************************************************************
 * CUBLAS Test execution
 ******************************************************************************/

/// CUBLAS handle
bool cublas_inited = false;
cublasHandle_t g_cublas_handle;


/**
 * Compute C = (alpha * A * B) + (beta * C)
 */
template <
    typename                    test_func_t,    ///< Test function type
    matrix_transform_t::kind_t  TransformA,     ///< Transformation op for matrix A
    matrix_transform_t::kind_t  TransformB,     ///< Transformation op for matrix B
    typename                    value_t,        ///< Multiplicand value type (matrices A and B)
    typename                    accum_t>        ///< Accumulator value type (matrix C and scalars)
bool test_cublas(
    value_t* A_data,
    value_t* B_data,
    value_t* C_data,
    int m,          ///< Height of C in rows
    int n,          ///< Width of C in columns
    int k,          ///< Width (height) of A (B)
    accum_t alpha,  ///< Multiplicand scalar
    accum_t beta)   ///< Addend scalar
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
        C_data,
        alpha,
        beta,
        stream,
        false).result;

    return error;
}

torch::Tensor blocksparse_matmul_transpose_cublas(torch::Tensor dense_a,
								      torch::Tensor dense_b,
								      torch::Tensor dense_out)
{
	typedef float       value_t;
	typedef float       accum_t;
	const math_operation_class_t math_op = math_operation_class_t::scalar;
// Define transpose constants
    static const matrix_transform_t::kind_t TransformA = matrix_transform_t::Transpose;
//    static const matrix_transform_t::kind_t TransformA = matrix_transform_t::NonTranspose;

//    static const matrix_transform_t::kind_t TransformB = matrix_transform_t::Transpose;
    static const matrix_transform_t::kind_t TransformB = matrix_transform_t::NonTranspose;

    auto sizes_a = dense_a.sizes().vec();
    auto sizes_b = dense_b.sizes().vec();
    auto sizes_out = dense_out.sizes().vec();

	// Initialize cuBLAS
	if (!cublas_inited) {
		if (cublasCreate(&g_cublas_handle) != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "cublasCreate() failed\n");
			exit(1);
		}
		cublas_inited = true;
	}

    value_t* A_data = (value_t*)dense_a.data_ptr();
    value_t* B_data = (value_t*)dense_b.data_ptr();
    value_t* C_data = (value_t*)dense_out.data_ptr();

    int m = sizes_a[0];
    int n = sizes_b[0];
    int k = sizes_b[1];

    //printf("m=%d, n=%d, k=%d\n", m, n, k);

    float alpha = 1.0;
    float beta = 0.0;



    bool test = test_cublas<
        cublas_gemm<gemm::tiling_strategy::Unknown, math_op, TransformA, TransformB, value_t, accum_t>,
        TransformA,
        TransformB,
        value_t,
        accum_t>(A_data, B_data, C_data, m, n, k, accum_t(alpha), accum_t(beta));


/*
     AT_DISPATCH_FLOATING_TYPES(dense_out.scalar_type(), "blocksparse_matmul_cuda", ([&] {


       blocksparse_matmul_transpose_kernel<scalar_t><<<blocks, threads_per_block>>>(
           dense_a.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
           sizes_a[0], sizes_a[1],
           row_ends_b.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
           cols_b.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
           data_b.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
           block_size_rows_b, block_size_cols_b,
           dense_out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()
           );
            }));
*/

    return dense_out;
}
