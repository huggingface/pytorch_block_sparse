import torch
import torch.nn
from . import block_sparse

class BlockSparseMatrixEmulator(block_sparse.BlockSparseMatrixBase):
    # cols is a list of nonzero block column indexes (int32)
    # row_start is a index into cols (int32)
    # Data is (len(cols), block_shape, block_shape)
    def __init__(self, shape, block_mask, data, block_shape):
        super(BlockSparseMatrixEmulator, self).__init__(shape, block_mask, data, block_shape)

    def rebuild(self, block_mask, block_ptr=None):
        super().rebuild(block_mask, block_ptr)
        self._dense = self.to_dense()
