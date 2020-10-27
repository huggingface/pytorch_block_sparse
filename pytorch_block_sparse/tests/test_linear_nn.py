import unittest
from unittest import TestCase

import torch
import torch.optim as optim

from pytorch_block_sparse import (
    BlockSparseLinear,
    BlockSparseMatrix,
    BlockSparseMatrixEmulator,
)
from pytorch_block_sparse.block_sparse_linear import PseudoBlockSparseLinear


class TestFun(TestCase):
    def test0(self):
        d = dict
        tests = [
            d(size_a=[8, 4], size_b=[4, 4], block_shape_b=(4, 1), density=0.5),
            d(size_a=[32, 32], size_b=[32, 32], block_shape_b=(4, 8), density=1.0),
            d(size_a=[32, 32], size_b=[32, 32], density=1.0),
            d(size_a=[256, 32], size_b=[32, 32], density=1.0),
        ]
        verbose = False
        for test in tests:
            lr = 0.001

            stride = 1
            size_a = test["size_a"]
            size_b = test["size_b"]
            block_shape_b = test.get("block_shape_b", (32, 32))
            # print(f"size_a={size_a}, size_b={size_b}")
            # Create the sparse linear layer
            linear = BlockSparseLinear(size_b[0], size_b[1], True, test["density"], block_shape=block_shape_b)
            if verbose:
                print(f"linear weight {linear.weight.data.shape}\n", linear.weight.data[::stride, ::stride])
                if hasattr(linear.weight, "_dense"):
                    print(
                        f"linear weight dense {linear.weight._dense.shape}\n", linear.weight._dense[::stride, ::stride]
                    )

            # TODO : this does nothing
            linear.cuda()

            # Input vector
            a1 = torch.nn.Parameter(torch.ones([size_a[0], size_a[1]]).cuda())
            a2 = torch.nn.Parameter(torch.ones([size_a[0], size_a[1]]).cuda())

            # Build a dense equivalent to the sparse
            dense = torch.nn.Parameter(linear.weight.to_dense().cuda())
            bias = torch.nn.Parameter(torch.zeros(size_b[1]).cuda())
            if verbose:
                print("dense\n", dense[::stride, ::stride])

            optimizer0 = optim.Adam([a1] + list(linear.parameters()), lr=lr)
            optimizer1 = optim.Adam([a2, dense, bias], lr=lr)

            for i in range(40):
                s = dense.isclose(linear.weight.to_dense(), atol=1e-05).all()

                if not s:
                    raise Exception("Matrices are different")

                optimizer0.zero_grad()
                optimizer1.zero_grad()

                # Apply the linear function
                b1 = linear(a1)

                # Compute a reference value
                b2 = a2.matmul(dense.t()) + bias

                # Check that both results match
                s = b1.isclose(b2, atol=1e-05).all()

                if not s:
                    raise Exception("Output are differents")

                loss1 = b1.sum()
                loss2 = b2.sum()

                loss1.backward()
                loss2.backward()

                s = a1.grad.isclose(a2.grad, atol=1e-05).all()
                if not s:
                    raise Exception("Input gradients are differents")

                a_grad = linear.weight.reverse_matmul(torch.ones_like(a1), transpose=False)

                s = a_grad.isclose(a2.grad, atol=1e-05).all()
                if not s:
                    print("input gradient 0\n", a_grad[::stride, ::stride])
                    print("input gradient 1\n", a1.grad[::stride, ::stride])
                    print("input gradient 2\n", a2.grad[::stride, ::stride])

                    raise Exception("Input gradients are differents, manual check")

                if verbose:
                    print("a_grad\n", a_grad[::stride, ::stride])
                    print("a1 grad\n", a1.grad[::stride, ::stride])
                    print("a2 grad\n", a2.grad[::stride, ::stride])

                    print(linear.weight.get_differentiable_data().grad)

                if isinstance(linear.weight, BlockSparseMatrix):
                    dense_grad = linear.weight.to_dense(data_replace=linear.weight.get_differentiable_data().grad)
                elif isinstance(linear.weight, BlockSparseMatrixEmulator):
                    dense_grad = linear.weight.get_differentiable_data().grad
                else:
                    raise RuntimeError("Unknown linear weight type {linear.weight.__class__}")
                dense_mask = linear.weight.to_dense(
                    data_replace=torch.ones_like(linear.weight.get_differentiable_data().grad)
                )

                dense_grad_reference = dense.grad * dense_mask

                if verbose:
                    print("dense_grad\n", dense_grad[::stride, ::stride])
                    print(
                        "dense_grad_reference\n",
                        dense_grad_reference[::stride, ::stride],
                    )

                s = dense_grad.isclose(dense_grad_reference, atol=1e-05).all()

                if not s:
                    raise Exception("Weight gradients are differents")

                optimizer0.step()
                optimizer1.step()

                with torch.no_grad():
                    dense *= dense_mask

    def test_pseudo_sparse(self):
        tests = [{"size_a": [256, 64], "size_b": [64, 128], "density": 1.0}]
        for test in tests:
            size_a = test["size_a"]
            size_b = test["size_b"]
            print(f"size_a={size_a}, size_b={size_b}")
            # Create the sparse linear layer
            linear = BlockSparseLinear(size_b[0], size_b[1], True, test["density"])
            with torch.no_grad():
                linear.weight.data.copy_(linear.weight.data.abs())
            pseudo_linear = PseudoBlockSparseLinear(linear)

            a1 = torch.randn([size_a[0], size_a[1]]).cuda().abs()

            b1_l = linear(a1)

            b1_pl = pseudo_linear(a1)

            self.assertTrue(torch.isclose(b1_l, b1_pl, rtol=1e-5).all())


if __name__ == "__main__":
    unittest.main()
