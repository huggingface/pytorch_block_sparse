from unittest import TestCase
import torch
import unittest
from pytorch_block_sparse.block_sparse import BlockSparseMatrix
from torch.autograd import gradcheck


class TestFun(TestCase):
    def helper_(self, sizes, block_size, block_count = None, blocks = None, density = None, iterations = 1,
                non_contiguous_a = False, non_contiguous_b = False):
        device = "cuda"
        a = torch.randn((sizes[0], sizes[1]), device=device)
        b = torch.randn((sizes[0], sizes[2]), device=device)

        if non_contiguous_a:
            print("NON CONTIGUOUS A")
            a = a.t().contiguous().t()
        else:
            print("CONTIGUOUS A")

        if non_contiguous_b:
            print("NON CONTIGUOUS B")
            b = b.t().contiguous().t()
        else:
            print("CONTIGUOUS B")

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
                    c = b.t().mm(a)
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
#                print(c_dense)

                c0_ = c0 * (c_dense != 0)

#                print(c_dense.shape, c0_.shape)

                s = c_dense.isclose(c0_, atol=1e-02).all()

                if not s.item():
                    print("Comparison NOK : transposed_matmul issue for ", k)
                    print("max difference %s=" % t["kind"], (c_dense - c0_).abs().max())
                    t["comparison"] = False
                else:
                    print("Comparison OK for transposed_matmul for ", k)
                    print("max difference %s=" % t["kind"], (c_dense - c0_).abs().max())
                    t["comparison"] = True

        return results

    def helper(self, sizes, block_size, density, iterations, inner_iterations, block_count = None, blocks = None,
               non_contiguous_a = False, non_contiguous_b = False):
        flops = float(2 * sizes[0] * sizes[1] * sizes[2])

        report = {}
        for i in range(iterations):
            results = self.helper_(sizes, block_size, block_count=block_count, blocks=blocks, density=density, iterations=inner_iterations,
                                   non_contiguous_a = non_contiguous_a, non_contiguous_b = non_contiguous_b)

            if "pytorch" in results:
                pytorch_time = results["pytorch"]["elapsed"]
            else:
                pytorch_time = None

            for kind, d in results.items():
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

    def check(self, results, block_size, blocks, verbose = False):
        cutlass_result = results["cutlass"]["output"]
        pytorch_result = results["pytorch"]["output"]

        if verbose:
            #print(cutlass_result)

            stride = 4
            print("cutlass block[0][0]", cutlass_result.data[::stride, ::stride])
            print("pytorch blocks[0][0]", pytorch_result[::stride, ::stride])
        for i in range(cutlass_result.blocks.shape[0] // 2):
            #print("i=", i)
            b = cutlass_result.blocks[i * 2:i * 2+2].flip(0) * torch.tensor(block_size, device=cutlass_result.blocks.device)
            #print("block position", b)

            b_pytorch = pytorch_result[b[0]:b[0] + block_size[0], b[1]:b[1] + block_size[1]]

            b_cutlass = cutlass_result.data[i*32:i*32 + 32].t()
            #print("cutlass full block\n", b_cutlass)
            #print("pytorch extracted block\n", b_pytorch)

            compare = b_pytorch.isclose(b_cutlass, atol=0.1)
            torch.set_printoptions(profile="full")
            torch.set_printoptions(profile="default")
            #break
            if not compare.all().item():
                #print("error on : i = %d" % i)
                raise Exception("Comparison failed", blocks)


    def test0(self):
        size = 32
        sizes = [size * 2, size * 4, size * 8]
        block_size = (32, 32)

        block_tests = [[(0, 0)],[(0,1)], [(1,0)], [(1,0), (0,2)], [(1,0), (2,0), (3,0)]]
        for blocks in block_tests:
            for non_contiguous_a in [False, True]:
                for non_contiguous_b in [False, True]:
                    results = self.helper(sizes, block_size, density = None, blocks = blocks, iterations = 1, inner_iterations = 1,
                                          non_contiguous_a = non_contiguous_a, non_contiguous_b = non_contiguous_b)
                    self.check(results, block_size, blocks, verbose = False)
            #break

    def test1(self):
        size = 512
        sizes = [size * 16 * 8, size * 2, size * 4]

        density = 0.47
        density = 1.0

        block_size = (32, 32)
        iterations = 4
        inner_iterations = 10

        for non_contiguous_a in [False, True]:
            for non_contiguous_b in [False, True]:
                results = self.helper(sizes, block_size, density, iterations, inner_iterations, block_count = None,
                                      non_contiguous_a=non_contiguous_a, non_contiguous_b=non_contiguous_b)

                self.check(results, block_size, results["cutlass"]["output"].blocks)



if __name__ == '__main__':
    unittest.main()


