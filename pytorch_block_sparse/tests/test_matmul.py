from unittest import TestCase
from pytorch_block_sparse.block_sparse import BlockSparseMatrix
import torch
import unittest

class TestFun(TestCase):
    def test0(self):
        bsm = BlockSparseMatrix.rand((64, 128), 10, (16, 16))

        a = torch.randn((256, 64))

        print(a.shape, bsm.shape)

        r = bsm.reverse_matmul(a)

        print(r)

if __name__ == '__main__':
    unittest.main()


