from unittest import TestCase
from pytorch_block_sparse.block_sparse import BlockSparseMatrix
import unittest
import torch

class TestFun(TestCase):
    def test0(self):
        bsm = BlockSparseMatrix.rand((64, 64), 10, (16, 16))
        bsm.sanity_check()

    def test1(self):
        bsm = BlockSparseMatrix.rand((64, 64), 10, (16, 16))
        d = bsm.to_dense()
        bsm.check_with_dense(d)
        bsm.sanity_check()

if __name__ == '__main__':
    unittest.main()
