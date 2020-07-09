from unittest import TestCase
import torch
import unittest

class TestFun(TestCase):

    def helper(self, sizes, block_size, block_count = None, density = None, iterations = 1):
        a = torch.randn((sizes[0], sizes[1]), device="cuda")

        if block_count == None:
            total_block_count = sizes[1] * sizes[2] / block_size[0] / block_size[1]
            block_count = int(total_block_count * density)

        from pytorch_block_sparse.block_sparse import BlockSparseMatrix
        bsm = BlockSparseMatrix.rand((sizes[2], sizes[1]), block_count, block_size, device="cuda")
        dbsm = bsm.to_dense()

        bsm.check_with_dense(dbsm)


        timings = {}
        compare = {}
        for kind in ["pytorch", "cutlass"]: #, "cuda"]: #, "cutlass"]:
            timing = []
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            dbsm_t = dbsm.t().contiguous()

            start.record()
            for i in range(iterations):
                if kind == "pytorch":
                    c = a.matmul(dbsm_t)
                elif kind == "cuda":
                    c = bsm.transposed_matmul(a)
                elif kind == "cublas":
                    import block_sparse_native
                    prr = torch.zeros((sizes[2], sizes[0]), device="cuda")
                    prr = prr.t()
                    cs = block_sparse_native.blocksparse_matmul_transpose_dense(a, dbsm, prr)
                elif kind == "cutlass":
                    c = bsm.transposed_matmul(a, method = 1)

            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)

            timing = dict(kind = kind, elapsed = elapsed, result=c)
            timings[kind] = timing


        c0 = timings["pytorch"]["result"]
        for k, t in timings.items():
            c = t["result"]
            s = c.isclose(c0, atol=1e-03).all()
            if not s.item():
                print("Comparison failed : transposed_matmul issue")
                print("max difference %s=" % t["kind"], (c - c0).abs().max())
            else:
                print("Comparison ok for transposed_matmul")

        return timings



    def tst0(self):
        sizes = [64, 16, 32]
        block_size = (16,16)
        block_count = 2
        for i in range(2):
            time_sparse, time_dense = self.helper(sizes, block_size, block_count)
            if i != 0:
                print("time_sparse=%f, time_dense = %s" % (time_sparse, time_dense))

    def test1(self):
        #sizes = [8192, 512, 2048]
        sizes = [1024, 512, 512]
        #sizes = [64, 16, 32]

        flops = float(2 * sizes[0] * sizes[1] * sizes[2])

        block_size = (32, 32)
        iterations = 10
        for i in range(10):
            timings = self.helper(sizes, block_size, density = 1.0, iterations = iterations)

            pytorch_time = timings["pytorch"]["elapsed"]

            for kind, d in timings.items():
                kind = d["kind"]
                kind_elapsed = d["elapsed"]

                print("kind = %s, elapsed=%f, gflops = %f, ratio = %s" % (kind, kind_elapsed, flops * iterations / kind_elapsed / 1e6, kind_elapsed / pytorch_time))


if __name__ == '__main__':
    # Pytorch version
    #sizes = [64, 16, 32]
#    a = torch.randn((sizes[0], sizes[1]), device="cuda")
#    dbsm_f = torch.zeros(sizes[1], sizes[2], device="cuda")
#    c_pytorch = a.matmul(dbsm_f)
    unittest.main()


