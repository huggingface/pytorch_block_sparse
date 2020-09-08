from unittest import TestCase
import unittest
from pytorch_block_sparse import BlockSparseMatrix, MagnitudeSparseOptimizerStrategy

class TestFun(TestCase):
    def test0(self):
        bsm = BlockSparseMatrix.randn((256, 256), 32, block_shape=(32,32), device="cuda")

        strategy = MagnitudeSparseOptimizerStrategy(0.1)

        strategy.run(bsm)





if __name__ == '__main__':
    unittest.main()
