/******************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

/**
 * \file
 * Architecture-specific GEMM block_task policies
 */

#include <stdint.h>

#include "../util/util.h"
#include "block_task.h"
#include "grid_raster.h"

namespace cutlass {
namespace gemm {


/******************************************************************************
 * tiling_strategy
 ******************************************************************************/

/**
 * Enumeration of tile-sizing granularities
 */
struct tiling_strategy : printable_t
{
    /// \brief Enumerants
    enum kind_t
    {
        Unknown,
        Small,
        Medium,
        Large,
        Tall,
        Wide,
        Huge,
        Custom
    };

    /// Enumerant value
    kind_t kind;

    /// Default constructor
    tiling_strategy() : kind(Unknown) {}

    /// Copy constructor
    tiling_strategy(const kind_t &other_kind) : kind(other_kind) {}

    /// Cast to kind_t
    operator kind_t() const { return kind; }

    /// Returns the instance as a string
    __host__ __device__ inline
    char const* to_string() const
    {
        switch (kind)
        {
            case Small:     return "small";
            case Medium:    return "medium";
            case Large:     return "large";
            case Tall:      return "tall";
            case Wide:      return "wide";
            case Huge:      return "huge";
            case Custom:    return "Custom";
            case Unknown:
            default:        return "unknown";
        }
    }

    /// Insert the formatted instance into the output stream
    void print(std::ostream& out) const { out << to_string(); }
};


/******************************************************************************
 * GEMM
 ******************************************************************************/

/**
 * GEMM task policy specialization for sgemm
 */
template <
    typename value_t,
    typename accum_t,
    matrix_transform_t::kind_t TransformA,      ///< Transformation op for matrix A
    matrix_transform_t::kind_t TransformB,      ///< Transformation op for matrix B
    tiling_strategy::kind_t TilingStrategy>     ///< Tile-sizing classification
struct gemm_policy;


/******************************************************************************
 * SGEMM
 ******************************************************************************/

/**
 * GEMM task policy specialization for Custom sgemm
 */
template <
    matrix_transform_t::kind_t TransformA,      ///< Transformation op for matrix A
    matrix_transform_t::kind_t TransformB>      ///< Transformation op for matrix B
struct gemm_policy<float, float, TransformA, TransformB, tiling_strategy::Custom> :
    block_task_policy<
        32,     // _BlockItemsY
        32,     // _BlockItemsX
        32,      // _BlockItemsK
        4,      // _ThreadItemsY
        4,      // _ThreadItemsX
        false,  // _UseDoubleScratchTiles
        grid_raster_strategy::Default>   // _RasterStrategy
{};


/**
 * GEMM task policy specialization for small sgemm
 */
template <
    matrix_transform_t::kind_t TransformA,      ///< Transformation op for matrix A
    matrix_transform_t::kind_t TransformB>      ///< Transformation op for matrix B
struct gemm_policy<float, float, TransformA, TransformB, tiling_strategy::Small> :
    block_task_policy<
        16,     // _BlockItemsY
        16,     // _BlockItemsX
        16,     // _BlockItemsK
        2,      // _ThreadItemsY
        2,      // _ThreadItemsX
        false,  // _UseDoubleScratchTiles
        grid_raster_strategy::Default>   // _RasterStrategy
{};


/**
 * GEMM task policy specialization for medium sgemm
 */
template <
    matrix_transform_t::kind_t TransformA,      ///< Transformation op for matrix A
    matrix_transform_t::kind_t TransformB>      ///< Transformation op for matrix B
struct gemm_policy<float, float, TransformA, TransformB, tiling_strategy::Medium> :
    block_task_policy<
        32,     // _BlockItemsY
        32,     // _BlockItemsX
        8,      // _BlockItemsK
        4,      // _ThreadItemsY
        4,      // _ThreadItemsX
        false,  // _UseDoubleScratchTiles
        grid_raster_strategy::Default>   // _RasterStrategy
{};

/**
 * GEMM task policy specialization for large sgemm
 */
template <
    matrix_transform_t::kind_t TransformA,      ///< Transformation op for matrix A
    matrix_transform_t::kind_t TransformB>      ///< Transformation op for matrix B
struct gemm_policy<float, float, TransformA, TransformB, tiling_strategy::Large> :
    block_task_policy<
        64,     // _BlockItemsY
        64,     // _BlockItemsX
        8,      // _BlockItemsK
        8,      // _ThreadItemsY
        8,      // _ThreadItemsX
        false,  // _UseDoubleScratchTiles
        grid_raster_strategy::Default>   // _RasterStrategy
{};

/**
 * GEMM task policy specialization for tall sgemm
 */
template <
    matrix_transform_t::kind_t TransformA,      ///< Transformation op for matrix A
    matrix_transform_t::kind_t TransformB>      ///< Transformation op for matrix B
struct gemm_policy<float, float, TransformA, TransformB, tiling_strategy::Tall> :
    block_task_policy<
        128,    // _BlockItemsY
        32,     // _BlockItemsX
        8,      // _BlockItemsK
        8,      // _ThreadItemsY
        4,      // _ThreadItemsX
        false,  // _UseDoubleScratchTiles
        grid_raster_strategy::Default>   // _RasterStrategy
{};

/**
 * GEMM task policy specialization for wide sgemm
 */
template <
    matrix_transform_t::kind_t TransformA,      ///< Transformation op for matrix A
    matrix_transform_t::kind_t TransformB>      ///< Transformation op for matrix B
struct gemm_policy<float, float, TransformA, TransformB, tiling_strategy::Wide> :
    block_task_policy<
        32,     // _BlockItemsY
        128,    // _BlockItemsX
        8,      // _BlockItemsK
        4,      // _ThreadItemsY
        8,      // _ThreadItemsX
        false,  // _UseDoubleScratchTiles
        grid_raster_strategy::Default>   // _RasterStrategy
{};

/**
 * GEMM task policy specialization for huge sgemm
 */
template <
    matrix_transform_t::kind_t TransformA,      ///< Transformation op for matrix A
    matrix_transform_t::kind_t TransformB>      ///< Transformation op for matrix B
struct gemm_policy<float, float, TransformA, TransformB, tiling_strategy::Huge> :
    block_task_policy<
        128,    // _BlockItemsY
        128,    // _BlockItemsX
        8,      // _BlockItemsK
        8,      // _ThreadItemsY
        8,      // _ThreadItemsX
        false,  // _UseDoubleScratchTiles
        grid_raster_strategy::Default>   // _RasterStrategy
{};


/******************************************************************************
 * DGEMM
 ******************************************************************************/

/**
 * GEMM task policy specialization for Custom dgemm
 */
template <
    matrix_transform_t::kind_t TransformA,      ///< Transformation op for matrix A
    matrix_transform_t::kind_t TransformB>      ///< Transformation op for matrix B
struct gemm_policy<double, double, TransformA, TransformB, tiling_strategy::Custom> :
    block_task_policy<
        32,     // _BlockItemsY
        32,     // _BlockItemsX
        32,      // _BlockItemsK
        4,      // _ThreadItemsY
        4,      // _ThreadItemsX
        false,  // _UseDoubleScratchTiles
        grid_raster_strategy::Default>   // _RasterStrategy
{};

/**
 * GEMM task policy specialization for small dgemm
 */
template <
    matrix_transform_t::kind_t TransformA,      ///< Transformation op for matrix A
    matrix_transform_t::kind_t TransformB>      ///< Transformation op for matrix B
struct gemm_policy<double, double, TransformA, TransformB, tiling_strategy::Small> :
    block_task_policy<
        16,     // _BlockItemsY
        16,     // _BlockItemsX
        16,     // _BlockItemsK
        2,      // _ThreadItemsY
        2,      // _ThreadItemsX
        false,  // _UseDoubleScratchTiles
        grid_raster_strategy::Default>   // _RasterStrategy
{};


/**
 * GEMM task policy specialization for medium dgemm
 */
template <
    matrix_transform_t::kind_t TransformA,      ///< Transformation op for matrix A
    matrix_transform_t::kind_t TransformB>      ///< Transformation op for matrix B
struct gemm_policy<double, double, TransformA, TransformB, tiling_strategy::Medium> :
    block_task_policy<
        32,     // _BlockItemsY
        32,     // _BlockItemsX
        16,     // _BlockItemsK
        4,      // _ThreadItemsY
        4,      // _ThreadItemsX
        false,  // _UseDoubleScratchTiles
        grid_raster_strategy::Default>   // _RasterStrategy
{};

/**
 * GEMM task policy specialization for large dgemm
 */
template <
    matrix_transform_t::kind_t TransformA,      ///< Transformation op for matrix A
    matrix_transform_t::kind_t TransformB>      ///< Transformation op for matrix B
struct gemm_policy<double, double, TransformA, TransformB, tiling_strategy::Large> :
    block_task_policy<
        64,     // _BlockItemsY
        64,     // _BlockItemsX
        8,      // _BlockItemsK
        4,      // _ThreadItemsY
        4,      // _ThreadItemsX
        false,  // _UseDoubleScratchTiles
        grid_raster_strategy::Default>   // _RasterStrategy
{};

/**
 * GEMM task policy specialization for tall dgemm
 */
template <
    matrix_transform_t::kind_t TransformA,      ///< Transformation op for matrix A
    matrix_transform_t::kind_t TransformB>      ///< Transformation op for matrix B
struct gemm_policy<double, double, TransformA, TransformB, tiling_strategy::Tall> :
    block_task_policy<
        128,    // _BlockItemsY
        32,     // _BlockItemsX
        8,      // _BlockItemsK
        8,      // _ThreadItemsY
        4,      // _ThreadItemsX
        false,  // _UseDoubleScratchTiles
        grid_raster_strategy::Default>   // _RasterStrategy
{};

/**
 * GEMM task policy specialization for wide dgemm
 */
template <
    matrix_transform_t::kind_t TransformA,      ///< Transformation op for matrix A
    matrix_transform_t::kind_t TransformB>      ///< Transformation op for matrix B
struct gemm_policy<double, double, TransformA, TransformB, tiling_strategy::Wide> :
    block_task_policy<
        32,     // _BlockItemsY
        128,    // _BlockItemsX
        8,      // _BlockItemsK
        4,      // _ThreadItemsY
        8,      // _ThreadItemsX
        false,  // _UseDoubleScratchTiles
        grid_raster_strategy::Default>   // _RasterStrategy
{};

/**
 * GEMM task policy specialization for huge dgemm
 */
template <
    matrix_transform_t::kind_t TransformA,      ///< Transformation op for matrix A
    matrix_transform_t::kind_t TransformB>      ///< Transformation op for matrix B
struct gemm_policy<double, double, TransformA, TransformB, tiling_strategy::Huge> :
    block_task_policy<
        64,     // _BlockItemsY
        128,    // _BlockItemsX
        8,      // _BlockItemsK
        8,      // _ThreadItemsY
        8,      // _ThreadItemsX
        false,  // _UseDoubleScratchTiles
        grid_raster_strategy::Default>   // _RasterStrategy
{};


} // namespace gemm
} // namespace cutlass
