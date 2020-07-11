from unittest import TestCase
from pytorch_block_sparse.block_sparse import BlockSparseMatrix
import unittest
import torch

class TestFun(TestCase):
    def tst0(self):
        bsm = BlockSparseMatrix.rand((64, 64), 10, (16, 16))
        bsm.sanity_check()

    def tst1(self):
        bsm = BlockSparseMatrix.rand((64, 64), 10, (16, 16))
        d = bsm.to_dense()
        bsm.check_with_dense(d)
        bsm.sanity_check()

    def test2(self):
        from block_sparse_devtools import bsc_build_from_dense

        for i in range(10):
            nblocks = 4 * 64
            shape = (128 * 8, 256 * 8)
            block_shape = (32,32)
            bsm = BlockSparseMatrix.rand(shape, nblocks, block_shape)
            d = bsm.to_dense()

            blocks_count = bsm.blocks_count()
            data, ptr, indices = bsc_build_from_dense(blocks_count[0],
                                                      blocks_count[1],
                                                      nblocks,
                                                      block_shape[0],
                                                      block_shape[1],
                                                      d,
                                                      shape[1],
                                                      shape[0]
                                                      )

            th_data_size = block_shape[0] * block_shape[1] * nblocks

            assert((ptr == bsm.row_start_ends_a).all())
            assert((indices == bsm.cols_a[:,0]).all())

            assert((data == bsm.data.flatten()).all())
            assert(data.shape[0] == th_data_size)
            assert(data.min() == d.min())
            assert(data.max() == d.max())



#        print(f"data.shape={data.shape}, data[0]={data[0]}, data[-1]={data[-1]}, data.min()={data.min()}, data.max()={data.max()}")


if __name__ == '__main__':
    unittest.main()
