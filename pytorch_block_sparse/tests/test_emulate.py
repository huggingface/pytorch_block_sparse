from unittest import TestCase

import torch

from pytorch_block_sparse import BlockSparseMatrix, BlockSparseMatrixEmulator


class TestFun(TestCase):
    def help_contruct(self, shape, block_mask, data, block_shape=(16, 16)):
        try:
            real = BlockSparseMatrix(shape, block_mask, data, block_shape)
        except Exception:
            real = None
        emul = BlockSparseMatrixEmulator(shape, block_mask, data, block_shape)

        return real, emul

    def help_randn(
        cls,
        shape,
        n_blocks,
        blocks=None,
        block_shape=(32, 32),
        device="cuda",
        positive=False,
    ):
        try:
            real = BlockSparseMatrix.randn(shape, n_blocks, blocks, block_shape, device=device, positive=positive)
        except Exception:
            real = None
        emul = BlockSparseMatrixEmulator.randn(shape, n_blocks, blocks, block_shape, device=device, positive=positive)

        return real, emul

    def test0(self):
        d = dict
        test_sizes = [d(nb=2, s=(4, 8), bs=(1, 4))]
        map = d(nb="n_blocks", s="shape", bs="block_shape")

        for ts in test_sizes:
            ts = {map[k]: v for k, v in ts.items()}
            self.help_randn(**ts, device="cpu")

    def test_from_dense(self):
        dense = torch.randn(8, 8).cuda()
        d = dict

        tests = [
            d(blocks=[[0, 0], [1, 3], [3, 2]], block_shape=(1, 2)),
            d(blocks=[[0, 0], [1, 1], [3, 1]], block_shape=(2, 4)),
            d(block_shape=(2, 4)),
        ]

        for test in tests:
            blocks = test.get("blocks")
            nblocks = test.get("nblocks")
            block_shape = test["block_shape"]
            mask = BlockSparseMatrixEmulator.ones(
                dense.shape, block_shape=block_shape, blocks=blocks, n_blocks=nblocks
            ).to_dense()

            versions = []
            for slow in False, True:
                sparse = BlockSparseMatrixEmulator.from_dense(dense, block_shape=block_shape, blocks=blocks, slow=slow)
                versions.append(sparse)

            for i, sparse in enumerate(versions):
                dense2 = sparse.to_dense()
                self.assertTrue(((dense * mask) == dense2).all())
