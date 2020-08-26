from unittest import TestCase
import torch
import unittest
from pytorch_block_sparse.block_sparse import BlockSparseMatrix

class TestFun(TestCase):
    def helper(self, sizes, block_size, block_count = None, density = None, blocks = None, iterations = 1, device = "cuda", transpose = True, verbose = False):
        device = device
        if isinstance(sizes[0], tuple):
            sizes_0 = sizes[0]
        else:
            sizes_0 = (sizes[0],)

        if transpose:
            a = torch.randn(sizes_0 + (sizes[1],), device=device)
        else:
            a = torch.randn(sizes_0 + (sizes[2],), device=device)

        #torch.set_printoptions(precision=10, edgeitems=100000, linewidth=10000)
        if verbose:
            print("a=", a, "\n")

        if block_count == None and blocks == None:
            total_block_count = sizes[1] * sizes[2] / block_size[0] / block_size[1]
            block_count = int(total_block_count * density)

        bsm = BlockSparseMatrix.randn((sizes[2], sizes[1]), block_count, blocks = blocks, block_shape=block_size, device=device)
        dbsm = bsm.to_dense()
        if verbose:
            print("b=", dbsm, "\n")
            print("a.shape", a.shape)
        bsm.check_with_dense(dbsm)

        timings = {}
        for kind in ["pytorch", "cutlass"]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            for i in range(iterations):
                if kind == "pytorch":
                    if transpose:
                        dbsm_ = dbsm.t()
                    else:
                        dbsm_ = dbsm
                    c = a.matmul(dbsm_)

                    if verbose:
                        print("c=", c, "\n")

                elif kind == "cutlass":
                    c = bsm.reverse_matmul(a, transpose)
                elif kind == "cublas":
                    import block_sparse_native
                    prr = torch.zeros((sizes[2], sizes[0]), device=device)
                    prr = prr.t()
                    cs = block_sparse_native.blocksparse_matmul_transpose_dense(a, dbsm, prr)
                elif kind == "cuda":
                    c = bsm.matmul_cuda(a)

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
                #torch.set_printoptions(precision=8, edgeitems=100000, linewidth=10000)
                stride = 32
                shift = 0
                c_ = c[shift::stride,shift::stride]
                c0_ = c0[shift::stride,shift::stride]
                if verbose:
                    print("c shape", c.shape)
                    print("c\n",c_)
                    print("c0\n", c0_)
                    print("c!=0\n", (c_ != 0).long())
                    print("c0!=0\n", (c0_ != 0).long())
                    print("equals\n", ((c_ - c0_).abs() < 1e-06).long())
                    print("equals nonzero\n", ((c_ - c0_).abs() > 1e-06).nonzero())

                s = c.isclose(c0, atol=1e-05).all()
                if not s.item():
                    print("max difference %s=" % t["kind"], (c - c0).abs().max())
                    t["comparison"] = False
                    raise Exception("Comparison NOK : reversed_matmul issue for ", k)
                else:
                    if verbose:
                        print("Comparison OK for reversed_matmul for ", k)
                        print("max difference %s=" % t["kind"], (c - c0).abs().max())
                    t["comparison"] = True
                if verbose:
                    print("c_cutlass=", c)
        #torch.set_printoptions(profile="default")

        return timings


    def test0(self):
        tests = [{"sizes":[32, 32, 32],
                  "block_setups":[
                    [(0,0)],
                                  ]
                  },
                 {"sizes":[32, 64, 32],
                  "block_setups":[
                    [(0,0)],
                                  ]
                  },
                 {"sizes": [64, 32, 32],
                  "block_setups": [
                      [(0, 0)],
                  ]
                  },
                 {"sizes": [128, 32, 32],
                  "block_setups": [
                      [(0, 0)],
                  ]
                 },
                 {"sizes": [128, 64, 32],
                  "block_setups": [
                      [(0, 0)],
                      [(0, 1)],
                      [(0, 0), (0,1)],
                  ]
                  }
                 ]
        tests += [{"sizes": [32, 32, 64],
                  "block_setups": [
                      [(0,0)],
                      [(1, 0)],
                      [(0,0), (1,0)],
                  ]
                 }                 
                 ]
        tests += [{"sizes": [(64, 32), 32, 64],
                  "block_setups": [
                      [(0,0)],
                      [(1, 0)],
                      [(0,0), (1,0)],
                  ]
                 }
                 ]
        tests += [{"sizes": [(64, 128, 32), 128, 256],
                  "block_setups": [
                      [(0,0)],
                      [(1, 0)],
                      [(0,0), (1,0)],
                  ]
                 }
                 ]
        tests += [{"sizes": [32, 64, 64],
                   "block_setups": [
                       [(0, 0), (1,0), (0, 1)],
                   ]
                   }
                  ]
        tests += [{"sizes": [32, 64, 128],
                   "block_setups": [
                       [(0, 0), (1,0), (0,1), (2,0)],
                   ]
                   }
                  ]
        block_size = (32,32)
        device = "cuda"
        for transpose in [False, True]:
            for test_info in tests:
                sizes = test_info["sizes"]
                for blocks in test_info["block_setups"]:
                    print(sizes, blocks)
                    timings = self.helper(sizes, block_size, density = None, blocks = blocks, device =device, verbose= False, transpose = transpose)

    def test1(self):
        size = 512
        sizes = [(4 * size * 2, 16), size * 2, size * 4]
        #sizes = [(4 * 2, 16), size * 2, size * 4]
        #size = 32
        #sizes = [32, size * 2, size * 2]

        #density = 0.42
        density = 1.0

        import functools
        import operator
        if isinstance(sizes[0], int):
            sizes_0 = sizes[0]
        else:
            sizes_0 = functools.reduce(operator.mul, sizes[0], 1)
        flops = float(2 * sizes_0 * sizes[1] * sizes[2])

        block_size = (32, 32)
        iterations = 1

        results = {}
        for transpose in [False, True]:
            for i in range(1):
                timings = self.helper(sizes, block_size, density = density, iterations = iterations, verbose=False, transpose = transpose)

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
                    gflops = flops * iterations / kind_elapsed / 1e6
                    print(f"kind = {kind}, transpose = {transpose}, elapsed={kind_elapsed}, gflops = {gflops}, ratio = {ratio}")

            print(results)

if __name__ == '__main__':
    unittest.main()


