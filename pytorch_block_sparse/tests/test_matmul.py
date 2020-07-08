from unittest import TestCase
from pytorch_block_sparse.block_sparse import BlockSparseMatrix
import torch
import unittest
import block_sparse_native

class TestFun(TestCase):

    def helper(self, sizes, block_size, block_count = None, density = None, iterations = 1):
        if block_count == None:
            total_block_count = sizes[1] * sizes[2] / block_size[0] / block_size[1]
            block_count = int(total_block_count * density)

        bsm = BlockSparseMatrix.rand((sizes[2], sizes[1]), block_count, block_size, device="cuda")
        dbsm = bsm.to_dense()

        bsm.check_with_dense(dbsm)

        # Pytorch version
        a = torch.randn((sizes[0], sizes[1]), device="cuda")
        start0 = torch.cuda.Event(enable_timing=True)
        end0 = torch.cuda.Event(enable_timing=True)
        start1 = torch.cuda.Event(enable_timing=True)
        end1 = torch.cuda.Event(enable_timing=True)


        dbsm_t = dbsm.t().contiguous()

        start1.record()
        for i in range(iterations):
            c1 = a.matmul(dbsm_t)
        end1.record()
        torch.cuda.synchronize()
        time1 = start1.elapsed_time(end1)


        # CUBLAS version
        start2 = torch.cuda.Event(enable_timing=True)
        end2 = torch.cuda.Event(enable_timing=True)

        prr = torch.zeros((sizes[2], sizes[0]), device="cuda")
        prr = prr.t()

        start2.record()
        for i in range(iterations):
            print("a=", a.shape, "b=", dbsm.shape, "c=", prr.shape)
            c2 = block_sparse_native.blocksparse_matmul_transpose_dense(a, dbsm, prr)
        end2.record()
        torch.cuda.synchronize()
        time2 = start1.elapsed_time(end2)

        # CUDA version
        start0.record()
        for i in range(iterations):
            c0 = bsm.transposed_matmul(a)
        end0.record()
        torch.cuda.synchronize()
        time0 = start0.elapsed_time(end0)

        if c1 is not None:
            s = c1.isclose(c0, atol=1e-03).all()
            #print(c0[::256 * 4, ::256 // 4])
            #print(c1[::256 * 4, ::256 // 4])
            if not s.item():
                print("Comparison failed : transposed_matmul issue")
                print("max difference cuda=", (c1 - c0).abs().max())
            else:
                print("Comparison ok for transposed_matmul")

        if c2 is not None:
            s = c2.isclose(c1,atol=1e-03).all()
            if not s.item():
                print("Comparison failed : cublas issue")
                print(c1[::256 * 8,::256])
                print(c2[::256 * 8,::256])
                print("max difference cublas=", (c2 - c1).abs().max())
            else:
                print("Comparison ok for cublas")


        return time0, time1, time2

    def tst0(self):
        sizes = [64, 16, 32]
        block_size = (16,16)
        block_count = 2
        for i in range(2):
            time_sparse, time_dense = self.helper(sizes, block_size, block_count)
            if i != 0:
                print("time_sparse=%f, time_dense = %s" % (time_sparse, time_dense))

    def test1(self):
        sizes = [892, 512, 2048]

        flops = 2 * sizes[0] * sizes[1] * sizes[2]

        block_size = (16, 16)
        iterations = 1
        for i in range(10):
            time_sparse, time_dense, time_cublas = self.helper(sizes, block_size, density = 1.0, iterations = iterations)

            print("time_sparse=%f, time_dense = %s, time_cublas = %s, ratio sparse = %s, ratio cublas = %s" % (time_sparse, time_dense, time_cublas,
                                                                                            time_sparse / time_dense,
                                                                                            time_cublas / time_dense,
                                                                                            ))
            print("gflops sparse=%f, gflops dense = %s, gflops cublas= %s" % (flops * iterations / time_sparse / 1e6 ,
                                                                              flops * iterations / time_dense / 1e6,
                                                                              flops * iterations / time_cublas / 1e6))


if __name__ == '__main__':
    unittest.main()


