import torch
import unittest
from unittest import TestCase
import tempfile
from pytorch_block_sparse import BlockSparseMatrix, BlockSparseLinear

class TestFun(TestCase):
    def test0(self):
        sizes = [64, 64]
        block_size = (32, 32)
        block_count = 2
        bsm = BlockSparseMatrix.randn(sizes, block_count, blocks = None, block_shape=block_size, device="cuda")

        tf = tempfile.NamedTemporaryFile()
        try:
            torch.save(bsm, tf.name)

            bsm2 = torch.load(tf.name)

            self.assertTrue((bsm.to_dense() == bsm2.to_dense()).all())

        finally:
            tf.close()

    def test1(self):
        sizes = [256, 256]

        linear = BlockSparseLinear(sizes[0], sizes[1], True, 0.5)

        tf = tempfile.NamedTemporaryFile()
        try:
            torch.save(linear, tf.name)

            linear2 = torch.load(tf.name)

            self.assertTrue((linear.weight.to_dense() == linear2.weight.to_dense()).all())
            self.assertTrue((linear.bias == linear2.bias).all())
        finally:
            tf.close()

    def test2(self):
        sizes = [256, 256]

        linear = BlockSparseLinear(sizes[0], sizes[1], True, 0.5)

        tf = tempfile.NamedTemporaryFile()
        try:
            state_dict = linear.state_dict()
            torch.save(state_dict, tf.name)

            linear2 = BlockSparseLinear(sizes[0], sizes[1], True, 0.5)

            linear2.load_state_dict(torch.load(tf.name))

            self.assertTrue((linear.weight.to_dense() == linear2.weight.to_dense()).all())
            self.assertTrue((linear.bias == linear2.bias).all())
        finally:
            tf.close()

    def tst3(self):
        sizes = [256, 256]

        linear = BlockSparseLinear(sizes[0], sizes[1], True, 0.5)

        tf = tempfile.NamedTemporaryFile()
        try:
            state_dict = linear.state_dict()
            torch.save(state_dict, tf.name)

            linear2 = BlockSparseLinear(sizes[0], sizes[1], True, 1.0)

            linear2.load_state_dict(torch.load(tf.name))

            self.assertTrue((linear.weight.to_dense() == linear2.weight.to_dense()).all())
            self.assertTrue((linear.bias == linear2.bias).all())
        finally:
            tf.close()


if __name__ == '__main__':
    unittest.main()
