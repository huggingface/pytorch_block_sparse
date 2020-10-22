from unittest import TestCase

from pytorch_block_sparse.block_sparse import (
    BlockSparseMatrix,
    BlockSparseMatrixEmulator,
)


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
        test_sizes = [d(nb=2, s=(3, 5), bs=(1, 1))]
        map = d(nb="n_blocks", s="shape", bs="block_shape")

        for ts in test_sizes:
            ts = {map[k]: v for k, v in ts.items()}
            self.help_randn(**ts, device="cpu")
