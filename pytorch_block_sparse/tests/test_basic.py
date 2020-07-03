from unittest import TestCase
from pytorch_block_sparse.block_sparse import BlockSparseMatrix
import unittest

class TestFun(TestCase):
    def check(self, bsm):
        cols = bsm.cols
        row_end = bsm.row_end

        if len(cols.shape) != 1:
            raise Exception("cols should be unidimensional, not of shape %s" % cols.shape)
        if cols.dtype != torch.int32:
            raise Exception("cols should be int32, not of type %s" % cols.dtype)
        max_col = cols.max()
        if max_col > shape[1] / block_shape[1]:
            raise Exception("cols max element (%d) cannot be larger than shape[1]/block_shape[1] (%d)" % (max_col, shape[1] / block_shape[1]))

        self.cols = cols

        if len(row_end.shape) != 1:
            raise Exception("row_end should be unidimensional, not of shape %s" % row_end.shape)
        if row_end.shape[0] != shape[0] / block_shape[0]:
            raise Exception("row_end.shape[0] (%d) should be equal to shape[0]/block_shape[0] (%d)" % (row_end.shape[0], shape[0] / block_shape[0]))
        if row_end.dtype != torch.int32:
            raise Exception("row_end should be int32, not of type %s" % row_end.dtype)

        max_row_end = row_end.max()
        if max_row_end > self.cols.shape[0]:
            raise Exception("row_end max element (%d) cannot be larger than cols count (%d)" % (max_row_end, self.cols.shape[0]))
        last_row_end = row_end[-1]
        if last_row_end != self.cols.shape[0]:
            raise Exception("row_end last element (%d) should be equal to cols count (%d)" % (last_row_end, self.cols.shape[0]))
        self.row_end = row_end


    def tst0(self):
        bsm = BlockSparseMatrix.rand((64, 64), 10, (16, 16))

    def test1(self):
        bsm = BlockSparseMatrix.rand((64, 64), 10, (16, 16))
        d = bsm.to_dense()
        bsm.check_with_dense(d)

if __name__ == '__main__':
    unittest.main()
