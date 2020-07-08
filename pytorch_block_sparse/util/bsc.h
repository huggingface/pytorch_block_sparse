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
 * BSC data structure providing basic CPU-based algorithms and
 * operations that can be cloned and synchronized in GPU device memory
 */

#include <vector>
#include <fstream>
#include <stdlib.h> /* abs */

#include <cutlass/util/debug.h>
#include "../cutlass/util/matrix_transform.h"


namespace cutlass {

/**
 * \brief BSC data structure providing basic CPU-based algorithms and
 * operations that be synchronized with a GPU-based replica
 * BSC format stores column-size ptr, row indices, and data
 */
template <typename value_t>
struct bsc
{
    // Host value type (must be convertible to/from value_t)
    typedef value_t host_value_t;


    //-----------------------------------------------------------------------------
    // Data members
    //-----------------------------------------------------------------------------

private:

    // number of blocks in N dimension
    int _NBlocks;

    // number of non-zero blocks
    int _nonZeroBlocks;

    // Block size information
    int _BlockItemsN;
    int _BlockItemsK;
    int _BlockItems;

    /// Data array on host
    std::vector<host_value_t> _h_data;

    /// Ptr array on host
    std::vector<int> _h_ptr;

    /// Indices array on host
    std::vector<int> _h_indices;

    /// Clone of data array on GPU device
    value_t *_d_data;

    /// Clone of ptr array on GPU device
    int *_d_ptr;

    /// Clone of Indices array on GPU device
    int *_d_indices;

    /// GPU Device identifier that clone synchronizes with
    int _device_id;


public:

    //-----------------------------------------------------------------------------
    // Lifetime and synchronization
    //-----------------------------------------------------------------------------

    /**
     * Constructor: convert the input matrix into the BSC format.
     */
    bsc(
        int NBlocks,  ///< pruned bsc information
        int KBlocks,
        int nonZeroBlocks,
        int BlockItemsN,
        int BlockItemsK,
        host_value_t *matrix, ///< the matrix that we are going to encode
        int k,        ///< Height of the matrix in rows
        int n)        ///< Width of the matrix in columns
    :
        _NBlocks(NBlocks),
        _nonZeroBlocks(nonZeroBlocks),
        _BlockItemsN(BlockItemsN),
        _BlockItemsK(BlockItemsK),
        _d_data(NULL),
        _d_ptr(NULL),
        _d_indices(NULL),
        _device_id(0)
    {
        _BlockItems = _BlockItemsN * _BlockItemsK;

        // set each host vector and device memory size
        _h_data.resize(_BlockItems * _nonZeroBlocks, 0);
        _h_ptr.resize(_NBlocks + 1, 0);
        _h_indices.resize(_nonZeroBlocks, 0);
        CUDA_PERROR_EXIT(cudaMalloc((void ** )&_d_data, sizeof(value_t) * _BlockItems * _nonZeroBlocks));
        CUDA_PERROR_EXIT(cudaMalloc((void ** )&_d_ptr, sizeof(int) * (_NBlocks + 1)));
        CUDA_PERROR_EXIT(cudaMalloc((void ** )&_d_indices, sizeof(int) * _nonZeroBlocks));


        // encode the input matrix into BSC format
        // initialize the non-zero block counting to use as offset
        int countNonZero = 0;
        for (int i=0; i < NBlocks; i++){
            // update ptr offset
            _h_ptr[i+1] = _h_ptr[i];
            for (int j=0; j < KBlocks; j++){
                // index of the top left element of the block
                int startX = i * BlockItemsN;
                int startY = j * BlockItemsK;

                // check the value summation of the block
                int SumBlock = 0;
                for (int x=0; x < _BlockItemsN; x++){
                    for (int y=0; y < _BlockItemsK; y++){
                        // check if the region is in the matrix
                        int YIdx = y + startY;
                        int XIdx = x + startX;
                        if ((YIdx < k) && (XIdx < n)){
                            SumBlock += abs(matrix[YIdx + XIdx * k]);
                        }
                    }
                }

                // If the block is non-zero block, then encode the block
                if (SumBlock != 0){
                    // update _h_data
                    for (int x=0; x < _BlockItemsN; x++){
                        for (int y=0; y < _BlockItemsK; y++){
                            // check if the region is in the matrix
                            int YIdx = y + startY;
                            int XIdx = x + startX;
                            if ((YIdx < k) && (XIdx < n)){
                                _h_data[countNonZero * _BlockItems + y + x * _BlockItemsK] = matrix[YIdx + XIdx * k];
                            }
                        }
                    }
                    // update _h_ptr
                    _h_ptr[i+1] ++;

                    // update _h_indices
                    _h_indices[countNonZero] = j;   

                    // increament the countNonZero value for this non-zero block
                    countNonZero ++;
                }
            }
        }

        // get device id
        CUDA_PERROR_EXIT(cudaGetDevice(&_device_id));
    }

    /// Destructor
    ~bsc()
    {
        if (_d_data)
            CUDA_PERROR_EXIT(cudaFree(_d_data));
        if (_d_ptr)
            CUDA_PERROR_EXIT(cudaFree(_d_ptr));
        if (_d_indices)
            CUDA_PERROR_EXIT(cudaFree(_d_indices));
    }

    /**
     * Synchronize the GPU-based replica with the current host-based matrix data
     */
    void sync_device()
    {
        size_t data_bytes = _BlockItems * _nonZeroBlocks * sizeof(value_t);
        size_t ptr_bytes = (_NBlocks + 1) * sizeof(int);
        size_t indices_bytes = _nonZeroBlocks * sizeof(int);
        CUDA_PERROR_EXIT(cudaMemcpy(_d_data, &_h_data[0], data_bytes, cudaMemcpyHostToDevice));
        CUDA_PERROR_EXIT(cudaMemcpy(_d_ptr, &_h_ptr[0], ptr_bytes, cudaMemcpyHostToDevice));
        CUDA_PERROR_EXIT(cudaMemcpy(_d_indices, &_h_indices[0], indices_bytes, cudaMemcpyHostToDevice));
    }


    /**
     * Synchronize the host-based replica with the current GPU-based matrix data
     */
    void sync_host()
    {
        size_t data_bytes = _BlockItems * _nonZeroBlocks * sizeof(value_t);
        size_t ptr_bytes = (_NBlocks + 1) * sizeof(int);
        size_t indices_bytes = _nonZeroBlocks * sizeof(int);
        CUDA_PERROR_EXIT(cudaMemcpy(&_h_data[0], _d_data, data_bytes, cudaMemcpyDeviceToHost));
        CUDA_PERROR_EXIT(cudaMemcpy(&_h_ptr[0], _d_ptr, ptr_bytes, cudaMemcpyDeviceToHost));
        CUDA_PERROR_EXIT(cudaMemcpy(&_h_indices[0], _d_indices, indices_bytes, cudaMemcpyDeviceToHost));
    }


    /**
     * Get host data pointer
     */
    value_t* h_data()
    {
        return _h_data.data();
    }


    /**
     * Get host data pointer
     */
    value_t const* h_data() const
    {
        return _h_data.data();
    }


    /**
     * Get host ptr pointer
     */
    int* h_ptr()
    {
        return _h_ptr.data();
    }


    /**
     * Get host ptr pointer
     */
    int const* h_ptr() const
    {
        return _h_ptr.data();
    }


    /**
     * Get host indices pointer
     */
    int* h_indices()
    {
        return _h_indices.data();
    }


    /**
     * Get host indices pointer
     */
    int const* h_indices() const
    {
        return _h_indices.data();
    }


    /**
     * Get device data pointer
     */
    value_t const* d_data() const
    {
        return _d_data;
    }

    /**
     * Get device data pointer
     */
    value_t * d_data()
    {
        return _d_data;
    }


    /**
     * Get device ptr pointer
     */
    int const* d_ptr() const
    {
        return _d_ptr;
    }

    /**
     * Get device ptr pointer
     */
    int * d_ptr()
    {
        return _d_ptr;
    }


    /**
     * Get device indices pointer
     */
    int const* d_indices() const
    {
        return _d_indices;
    }

    /**
     * Get device indices pointer
     */
    int * d_indices()
    {
        return _d_indices;
    }

    //-----------------------------------------------------------------------------
    // save the data 
    //-----------------------------------------------------------------------------
    std::ostream & write_bsc(std::ostream &out)
    {
        // write data
        for (int i=0; i < _nonZeroBlocks; i++){
            for (int y=0; y < _BlockItemsK; y++){
                for (int x=0; x < _BlockItemsN; x++){
                    out << (x ? "," : "") << _h_data[i * _BlockItems + y + x * _BlockItemsK];
                }
                out << "\n";
            }
            out << "\n";
        }
        out << "\n";

        // write ptr
        for (int i=0; i < (_NBlocks + 1); i++){
            out << (i ? "," : "") << _h_ptr[i];
        }
        out << "\n";

        // write indices
        for (int i=0; i < _nonZeroBlocks; i++){
            out << (i ? "," : "") << _h_indices[i];
        }
        out << "\n";

        return out;
    }


};


} // namespace cutlass
