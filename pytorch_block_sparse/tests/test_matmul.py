from unittest import TestCase
from pytorch_block_sparse.block_sparse import BlockSparseMatrix
import torch
import unittest

class TestFun(TestCase):

    def helper(self, sizes, block_size, block_count = None, density = None, iterate = 1):
        if block_count == None:
            total_block_count = sizes[1] * sizes[2] / block_size[0] / block_size[1]
            block_count = int(total_block_count * density)

        bsm = BlockSparseMatrix.rand((sizes[2], sizes[1]), block_count, block_size, device="cuda")
        dbsm = bsm.to_dense()

        bsm.check_with_dense(dbsm)

        a = torch.randn((sizes[0], sizes[1]), device="cuda")
        start0 = torch.cuda.Event(enable_timing=True)
        end0 = torch.cuda.Event(enable_timing=True)

        start0.record()
        for i in range(iterate):
            r = bsm.transposed_matmul(a)
        end0.record()

        torch.cuda.synchronize()

        time0 = start0.elapsed_time(end0)

        start1 = torch.cuda.Event(enable_timing=True)
        end1 = torch.cuda.Event(enable_timing=True)

        dbsm_t = dbsm.t().contiguous()
        pr = None
        start1.record()
        for i in range(iterate):
            pr = a.matmul(dbsm_t)
        end1.record()

        torch.cuda.synchronize()
        time1 = start1.elapsed_time(end1)

        if pr is not None:
            s = pr.isclose(r).all()
            if not s.item():
                raise Exception("Comparison failed : transposed_matmul issue")
            else:
                print("Comparison ok")

        return time0, time1

    def test0(self):
        sizes = [64, 16, 32]
        block_size = (16,16)
        block_count = 2
        for i in range(2):
            time_sparse, time_dense = self.helper(sizes, block_size, block_count)
            if i != 0:
                print("time_sparse=%f, time_dense = %s" % (time_sparse, time_dense))

    def test1(self):
        sizes = [512, 512, 2048]
        block_size = (16,16)
        for i in range(10):
            time_sparse, time_dense = self.helper(sizes, block_size, density = 1.0, iterate = 1000)
            if i != 0:
                print("time_sparse=%f, time_dense = %s" % (time_sparse, time_dense))


if __name__ == '__main__':
    unittest.main()


