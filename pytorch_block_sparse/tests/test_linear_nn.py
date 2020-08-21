from unittest import TestCase
import torch
import unittest
from torch.autograd import gradcheck
from pytorch_block_sparse.block_sparse_linear import BlockSparseLinear

class TestFun(TestCase):
    def test0(self):
        tests = [{"size_a": [32, 32],
                  "size_b": [32, 32],
                  "density": 1.0
                  },
                 {"size_a": [256, 32],
                  "size_b": [32, 32],
                  "density": 1.0
                  }
                 ]
        verbose = False
        for test in tests:
            stride = 8
            size_a = test["size_a"]
            size_b = test["size_b"]
            print(f"size_a={size_a}, size_b={size_b}")
            # Create the sparse linear layer
            linear = BlockSparseLinear(size_b[0], size_b[1], False, test["density"])
            if verbose:
                print(linear.weight_data[::stride,::stride])

            # TODO : this does nothing
            linear.cuda()

            # Input vector
            a1 = torch.nn.Parameter(torch.ones([size_a[0], size_a[1]]).cuda())
            a2 = torch.nn.Parameter(torch.ones([size_a[0], size_a[1]]).cuda())

            # Build a dense equivalent to the sparse
            dense = torch.nn.Parameter(linear.weight.to_dense())
            if verbose:
                print("dense\n", dense[::stride, ::stride])

            # Apply the linear function
            b1 = linear(a1)


            # Compute a reference value
            b2 = a2.matmul(dense.t())

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

            a_grad = linear.weight.transposed_reverse_matmul(torch.ones_like(a1), transpose=False)

            s = a_grad.isclose(a2.grad, atol=1e-05).all()
            if not s:
                raise Exception("Input gradients are differents, manual check")

            if verbose:
                print("a_grad\n", a_grad[::stride, ::stride])
                print("a1 grad\n", a1.grad[::stride,::stride])
                print("a2 grad\n", a2.grad[::stride,::stride])

                print(linear.weight_data.grad)

            dense_grad = linear.weight.to_dense(data_replace=linear.weight_data.grad)
            dense_mask = linear.weight.to_dense(data_replace=torch.ones_like(linear.weight_data.grad))

            dense_grad_reference = dense.grad * dense_mask

            if verbose:
                print("dense_grad_reference\n", dense_grad_reference[::stride,::stride])

            s = dense_grad.isclose(dense_grad_reference, atol=1e-05).all()

            if not s:
                raise Exception("Weight gradients are differents")





if __name__ == '__main__':
    unittest.main()




