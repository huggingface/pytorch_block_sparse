from unittest import TestCase
import torch
import unittest
from pytorch_block_sparse.block_sparse import BlockSparseMatrix
from torch.autograd import gradcheck


class TestFun(TestCase):
    def helper(self, sizes, block_size, block_count = None, density = None, iterations = 1):
        device = "cuda"
        a = torch.randn((sizes[0], sizes[1]), device=device)
        b = torch.randn((sizes[0], sizes[2]), device=device)

        if block_count == None:
            total_block_count = sizes[1] * sizes[2] / block_size[0] / block_size[1]
            block_count = int(total_block_count * density)

        bsm = BlockSparseMatrix.zero((sizes[2], sizes[1]), block_count, block_size, device=device)
        dbsm = bsm.to_dense()
        bsm.check_with_dense(dbsm)

        timings = {}

        for kind in ["pytorch", "cutlass"]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            result = None
            for i in range(iterations):
                if kind == "pytorch":
                    c = b.t().mm(a)
                    result = c
                elif kind == "cutlass":
                    c = bsm.matmul_support(b.t().contiguous(), a)
#                    dbsm2 = bsm.to_dense()

                    print("c.data.shape", c.data.shape)
                    result = c.data.t()
                    #for r in c.data:
                    #    print(r)
                    #c = c.to_dense()
#                    print("C data", c.data)

            print("C ", kind, result)

            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)

            timing = dict(kind = kind, elapsed = elapsed, result=c)
            timing["result"] = result
            timings[kind] = timing

        if "pytorch" in timings:
            c0 = timings["pytorch"]["result"]
            for k, t in timings.items():
                if k == "pytorch":
                    t["comparison"] = True
                    continue
                c = t["result"]
                continue

                s = c.isclose(c0, atol=1e-03).all()
                if not s.item():
                    print("Comparison NOK : transposed_matmul issue for ", k)
                    print("max difference %s=" % t["kind"], (c - c0).abs().max())
                    t["comparison"] = False
                else:
                    print("Comparison OK for transposed_matmul for ", k)
                    print("max difference %s=" % t["kind"], (c - c0).abs().max())
                    t["comparison"] = True

        r = (timings["pytorch"]["result"] - timings["cutlass"]["result"]).abs() < 0.0001
        torch.set_printoptions(profile="full")
        print("matching")
        #print(r.long())
        print(r.all())
        torch.set_printoptions(profile="default")

        print("output shape", timings["pytorch"]["result"].shape)

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
        size = 32
        sizes = [size * 2, size * 4, size * 8]
        print("size", sizes)
        density = 1.0

        flops = float(2 * sizes[0] * sizes[1] * sizes[2])

        block_size = (32, 32)
        iterations = 1
        inner_iterations = 1

        results = {}
        for i in range(iterations):
            timings = self.helper(sizes, block_size, density = density, iterations = inner_iterations)

            if "pytorch" in timings:
                pytorch_time = timings["pytorch"]["elapsed"]
            else:
                pytorch_time = None

            for kind, d in timings.items():
                if kind not in results:
                    results[kind] = {True:0, False:0}
                if "comparison" in d:
                    results[kind][d["comparison"]] += 1

                kind = d["kind"]
                kind_elapsed = d["elapsed"]
                if pytorch_time == None:
                    ratio = "Unknown"
                else:
                    ratio = kind_elapsed / pytorch_time

                print("kind = %s, elapsed=%f, gflops = %f, ratio = %s" % (kind, kind_elapsed, flops * inner_iterations / kind_elapsed / 1e6, ratio))
        print(results)

if __name__ == '__main__':
    unittest.main()


