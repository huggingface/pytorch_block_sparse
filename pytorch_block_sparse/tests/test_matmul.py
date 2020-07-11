from unittest import TestCase
import torch
import unittest

class TestFun(TestCase):

    def helper_old(self, sizes, block_size, block_count = None, density = None, iterations = 1):
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
        for kind in ["cutlass"]: #""pytorch", "cutlass"]: #, "cuda"]: #, "cutlass"]:
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

        if "pytorch" in timings:
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

    def helper(self, sizes, block_size, block_count = None, density = None, iterations = 1):
        a = torch.randn((sizes[0], sizes[1]), device="cuda")

        if block_count == None:
            total_block_count = sizes[1] * sizes[2] / block_size[0] / block_size[1]
            block_count = int(total_block_count * density)

        block_count = 1

        from pytorch_block_sparse.block_sparse import BlockSparseMatrix
        bsm = BlockSparseMatrix.rand((sizes[1], sizes[2]), block_count, block_size, device="cuda")
        dbsm = bsm.to_dense()
        dbsm_t = dbsm.t().contiguous()
        print("dbsm")
        print(dbsm[::32,::32])

        bsm.check_with_dense(dbsm)

        timings = {}
        compare = {}
        for kind in ["cutlass", "pytorch"]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            for i in range(iterations):
                if kind == "pytorch":
                    c = a.matmul(dbsm_t)
                    #print()
                    #print("a shape=", a.shape, "dbsm.shape=", dbsm.shape, "c.shape", c.shape)
                    #print("a=\n", a[::32,::32], "\ndbsm=\n", dbsm[::32,::32], "\nc=\n", c[::32, ::32])
                elif kind == "cutlass":
                    c = bsm.matmul(a)

            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)

            timing = dict(kind = kind, elapsed = elapsed, result=c)
            timings[kind] = timing

        if "pytorch" in timings:
            c0 = timings["pytorch"]["result"]
            for k, t in timings.items():
                if k == "pytorch":
                    t["comparison"] = True
                    continue
                c = t["result"]
                s = c.isclose(c0, atol=1e-03).all()
                if not s.item():
                    print("Comparison NOK : transposed_matmul issue for ", k)
                    print("max difference %s=" % t["kind"], (c - c0).abs().max())

                    #print(c[::16].abs().sum(1))
                    differences = ((c - c0).abs() > 0.01).int()
                   # print(differences.nonzero().shape)

                   #print(differences[::16].sum(1))

                    #print("differences = ", differences.int().sum().item())

                    #print(c[::16,::16])
                    #print(c0[::16,::16])

                    print("dbsm")
                    print(dbsm)

                    print("C")
                    print("c max", c.max())

                    cs = c[::32,::32]
                    cs0 = c0[::32,::32]

                    print("differences")
                    print(((cs - cs0).abs() > 0.1).int())

                    print("mask")
                    print(bsm.block_mask.int())

                    print("C")
                    for i, r in enumerate(cs):
                        print(",".join(map(lambda  x : "%02d" % x, r.int().cpu().numpy())))

                    print("")
                    print("C0")
                    for i, r in enumerate(cs0):
                         print(",".join(map(lambda x: "%02d" % x, r.int().cpu().numpy())))


                    t["comparison"] = False
                else:
                    print("Comparison OK for transposed_matmul for ", k)
                    print("max difference %s=" % t["kind"], (c - c0).abs().max())
                    t["comparison"] = True

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
        #sizes = [2048, 2048, 2048]
        sizes = [128 * 2, 128 * 2, 128 * 2]
        print(sizes[0] * sizes[2])
        #sizes = [64, 16, 32]

        flops = float(2 * sizes[0] * sizes[1] * sizes[2])

        block_size = (32, 32)
        iterations = 1

        results = {}
        for i in range(1):
            timings = self.helper(sizes, block_size, density = 1.0, iterations = iterations)

            if "pytorch" in timings:
                pytorch_time = timings["pytorch"]["elapsed"]
            else:
                pytorch_time = None

            for kind, d in timings.items():
                #if kind == "pytorch":
                #continue
                if kind not in results:
                    results[kind] = {True:0, False:0}
                results[kind][d["comparison"]] += 1

                kind = d["kind"]
                kind_elapsed = d["elapsed"]
                if pytorch_time == None:
                    ratio = "Unknown"
                else:
                    ratio = kind_elapsed / pytorch_time

                print("kind = %s, elapsed=%f, gflops = %f, ratio = %s" % (kind, kind_elapsed, flops * iterations / kind_elapsed / 1e6, ratio))
        print(results)

if __name__ == '__main__':
    # Pytorch version
    #sizes = [64, 16, 32]
#    a = torch.randn((sizes[0], sizes[1]), device="cuda")
#    dbsm_f = torch.zeros(sizes[1], sizes[2], device="cuda")
#    c_pytorch = a.matmul(dbsm_f)
    unittest.main()


