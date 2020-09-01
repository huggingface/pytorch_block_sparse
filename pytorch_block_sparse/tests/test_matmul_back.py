from unittest import TestCase
import torch
import unittest
from pytorch_block_sparse import BlockSparseMatrix

class TestFun(TestCase):
    def helper_(self, sizes, block_size, block_count = None, blocks = None, density = None, iterations = 1,
                non_contiguous_a = False, non_contiguous_b = False):
        device = "cuda"

        if isinstance(sizes[0], tuple):
            sizes_0 = sizes[0]
        else:
            sizes_0 = (sizes[0],)

        # Build positive matrices to easily check results
        a = torch.randn(sizes_0 + (sizes[1],), device=device).abs()
        b = torch.randn(sizes_0 + (sizes[2],), device=device).abs()

        if non_contiguous_a:
            a = a.transpose(-2, -1).contiguous().transpose(-2, -1)

        if non_contiguous_b:
            b = b.transpose(-2, -1).contiguous().transpose(-2, -1)

        if block_count == None and blocks == None:
            total_block_count = sizes[1] * sizes[2] / block_size[0] / block_size[1]
            block_count = int(total_block_count * density)

        bsm = BlockSparseMatrix.zeros((sizes[2], sizes[1]), block_count, blocks, block_size, device=device)

        results = {}

        kinds = ["pytorch", "cutlass"]
        kinds.reverse()
        for kind in kinds:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for i in range(iterations):
                if kind == "pytorch":
                    aa = a.reshape(-1, a.shape[-1])
                    bb = b.reshape(-1, b.shape[-1])
                    bb = bb.t()
                    c = bb.mm(aa)
                elif kind == "cutlass":
                    bsm.matmul_with_output_sparse_support(b, a, overwrite_data = True)
                    c = bsm


            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)

            result = dict(kind = kind, elapsed = elapsed, output=c)
            results[kind] = result

        if "pytorch" in results:
            c0 = results["pytorch"]["output"]


            for k, t in results.items():
                if k == "pytorch":
                    t["comparison"] = True
                    continue
                c = t["output"]

                c_dense = c.to_dense()

                c0_ = c0 * (c_dense != 0)

                s = c_dense.isclose(c0_, rtol=1e-4).all()

                if not s.item():
                    print("max difference %s=" % t["kind"], float((c_dense - c0_).abs().max()), float(c.data.abs().max()))
                    raise Exception("Comparison NOK : matmul_with_output_sparse_support issue for ", k)
                    t["comparison"] = False
                else:
                    #print("Comparison OK for matmul_with_output_sparse_support for ", k)
                    #print("max difference %s=" % t["kind"], float((c_dense - c0_).abs().max()))
                    t["comparison"] = True

        return results

    def helper(self, sizes, block_size, density, iterations, inner_iterations, block_count = None, blocks = None,
               non_contiguous_a = False, non_contiguous_b = False):

        import functools
        import operator

        if isinstance(sizes[0], int):
            sizes_0 = sizes[0]
        else:
            sizes_0 = functools.reduce(operator.mul, sizes[0], 1)

        flops = float(2 * sizes_0 * sizes[1] * sizes[2])

        report = {}
        for i in range(iterations):
            results = self.helper_(sizes, block_size, block_count=block_count, blocks=blocks, density=density, iterations=inner_iterations,
                                   non_contiguous_a = non_contiguous_a, non_contiguous_b = non_contiguous_b)

            if "pytorch" in results:
                pytorch_time = results["pytorch"]["elapsed"]
            else:
                pytorch_time = None

            for kind, d in results.items():
                if kind == "pytorch":
                    continue
                if kind not in report:
                    report[kind] = {True: 0, False: 0}
                if "comparison" in d:
                    report[kind][d["comparison"]] += 1

                kind = d["kind"]
                kind_elapsed = d["elapsed"]
                if pytorch_time == None:
                    ratio = "Unknown"
                else:
                    ratio = kind_elapsed / pytorch_time

                print("kind = %s, elapsed=%f, gflops = %f, ratio = %s" % (
                kind, kind_elapsed, flops * inner_iterations / kind_elapsed / 1e6, ratio))

        return results

    def check(self, results, sizes, block_size, blocks, verbose = False):
        if isinstance(sizes[0], tuple):
            sizes_0 = 1
            for s in sizes[0]:
                sizes_0 *= s
        else:
            sizes_0 = sizes[0]

        cutlass_result = results["cutlass"]["output"]
        pytorch_result = results["pytorch"]["output"]

        if verbose:
            #print(cutlass_result)

            stride = 4
            print("cutlass block[0][0]", cutlass_result.data[::stride, ::stride].t())
            print("pytorch blocks[0][0]", pytorch_result[::stride, ::stride])
        for i in range(cutlass_result.blocks.shape[0] // 2):
            b = cutlass_result.blocks[i * 2:i * 2+2].flip(0) * torch.tensor(block_size, device=cutlass_result.blocks.device)
            b_pytorch = pytorch_result[b[0]:b[0] + block_size[0], b[1]:b[1] + block_size[1]]

            b_cutlass = cutlass_result.data[i*32:i*32 + 32].t()

            compare = b_pytorch.isclose(b_cutlass, rtol=1e-4)
            if not compare.all().item():
                rel_diff = ((b_pytorch-b_cutlass).abs() / (1e-9 + b_pytorch.abs())).abs().max()
                max_diff = (b_pytorch-b_cutlass).abs().max()
                print(f"rel diff={rel_diff}, max diff={max_diff}, max_pytorch={b_pytorch.abs().max()}, max_cutlass={b_cutlass.abs().max()}")
                raise Exception(f"Comparison failed out_shape={cutlass_result.shape} blocks={blocks} sizes={sizes}")


    def test0(self):
        bsize = 32
        tests = [dict(sizes = [bsize * 2, bsize * 4, bsize * 8],
                      block_tests=[[(0, 0)] ,[(0,1)], [(1,0)], [(1,0), (0,2)], [(1,0), (2,0), (3,0)]]),
                 dict(sizes = [1, bsize, bsize], block_tests=[[(0, 0)]])
                 ]
        block_size = (32, 32)

        for test in tests:
            sizes = test["sizes"]
            blocks_tests = test["block_tests"]
            for blocks in blocks_tests:
                for non_contiguous_a in [False, True]:
                    for non_contiguous_b in [False, True]:
                        results = self.helper(sizes, block_size, density = None, blocks = blocks, iterations = 1, inner_iterations = 1,
                                              non_contiguous_a = non_contiguous_a, non_contiguous_b = non_contiguous_b)
                        self.check(results, sizes, block_size, blocks, verbose = False)

    def test1(self):
        size = 512

        test_sizes = [[(size * 16,  8), size * 2, size * 4],
                      [1, size * 2, size * 4],
                      ]
        test_densities = [1.0] #0.47, 1.0]

        block_size = (32, 32)
        iterations = 4
        inner_iterations = 10

        for sizes in test_sizes:
            for density in test_densities:
                for non_contiguous_a in [False, True]:
                    for non_contiguous_b in [False, True]:
                        results = self.helper(sizes, block_size, density, iterations, inner_iterations, block_count = None,
                                              non_contiguous_a=non_contiguous_a, non_contiguous_b=non_contiguous_b)
                        try:
                            self.check(results, sizes, block_size, results["cutlass"]["output"].blocks)
                        except:
                            raise Exception(
                                f"Comparison NOK : matmul_with_output_sparse_support issue for sizes={sizes}, density={density}, non_contiguous_a={non_contiguous_a}, non_contiguous_b={non_contiguous_b}")


if __name__ == '__main__':
    unittest.main()


