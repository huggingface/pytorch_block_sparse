from unittest import TestCase
import unittest
from pytorch_block_sparse import BlockSparseMatrix, SparseOptimizer, BlockSparseLinear
from pytorch_block_sparse.sparse_optimizer import MagnitudeSparseOptimizerStrategy
import torch
import torch.optim as optim

class TestFun(TestCase):
    def check_differences(self, bsm, reference_dense, expected_block_changes):
        dense = bsm.to_dense()

        differences = (reference_dense != dense)
        block_shape = bsm.block_shape
        differences = float(differences.float().sum() / (block_shape[0] * block_shape[1]))

        self.assertEqual(differences, expected_block_changes)

    def test0(self):
        size = (256, 256)
        block_count = 32
        cleanup_ratio = 0.1
        block_shape = (32,32)
        bsm = BlockSparseMatrix.randn(size, block_count, block_shape=block_shape, device="cuda")

        dense0 = bsm.to_dense()

        strategy = MagnitudeSparseOptimizerStrategy(cleanup_ratio)
        strategy.run(bsm)

        expected_block_changes = int(cleanup_ratio * block_count) * 2
        self.check_differences(bsm, dense0, expected_block_changes)

    def test_sparse_optimizer(self):
        size = (256, 256)
        block_count = 32
        cleanup_ratio = 0.1
        block_shape = (32, 32)
        bsm = BlockSparseMatrix.randn(size, block_count, block_shape=block_shape, device="cuda")
        dense0 = bsm.to_dense()

        so = SparseOptimizer([bsm], lr=cleanup_ratio)

        so.step()

        expected_block_changes = int(cleanup_ratio * block_count) * 2
        self.check_differences(bsm, dense0, expected_block_changes)

    def test_sparse_optimizer_attached_optimizer(self):
        size = (256, 256)
        density = 0.5
        cleanup_ratio = 0.1

        linear = BlockSparseLinear(size[0], size[1], True, density).cuda()

        sparse_objects = SparseOptimizer.sparse_objects(linear)

        self.assertEqual(len(sparse_objects), 1)

        so = SparseOptimizer(sparse_objects, lr=cleanup_ratio)

        adam = optim.Adam(linear.parameters())

        so.attach_optimizer(adam)

        # Run forward and backward
        a = torch.randn([1, size[0]]).abs().cuda()
        out = linear(a)

        loss = out.sum()

        loss.backward()

        adam.step()

        dense0 = linear.weight.to_dense()

        so.step()

        block_count = linear.block_count
        expected_block_changes = int(cleanup_ratio * block_count) * 2
        self.check_differences(linear.weight, dense0, expected_block_changes)






if __name__ == '__main__':
    unittest.main()
