from unittest import TestCase
from pytorch_block_sparse.block_sparse import BlockSparseMatrix
import torch
import unittest

class TestFun(TestCase):
    def test0(self):
        bsm = BlockSparseMatrix.rand((128, 64), 10, (16, 16))

        a = torch.randn((256, 64))

        r = bsm.transposed_matmul(a)

if __name__ == '__main__':
    unittest.main()


