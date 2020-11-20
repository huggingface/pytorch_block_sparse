import math
from typing import Tuple

import torch
import torch.autograd
import torch.nn as nn

from .block_sparse import (
    BlockSparseMatrix,
    BlockSparseMatrixBase,
    BlockSparseMatrixEmulator,
)


class BlockSparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_data, weight):
        check = False
        verbose = False

        if verbose or check:
            dense_weight = weight.to_dense()

        if verbose:
            stride = 8
            print("BlockSparseLinearFunction.forward input\n", input[::stride, ::stride])
            print(
                "BlockSparseLinearFunction.forward dense_weight\n",
                dense_weight[::stride, ::stride],
            )
            print(
                "BlockSparseLinearFunction.forward weight\n",
                weight.data[::stride, ::stride],
            )

        assert isinstance(weight, BlockSparseMatrixBase)

        ctx.save_for_backward(input, weight_data)
        ctx.weight = weight
        output = weight.reverse_matmul(input, transpose=True)
        if check:
            dense = weight.to_dense()
            output1 = input.matmul(dense.t())
            if not output1.isclose(output, ator=1e-05).all():
                raise Exception("BlockSparseLinearFunction.forward non matching output 1")
            else:
                if verbose:
                    print("BlockSparseLinearFunction.forward matching output 1")

        if verbose:
            print("BlockSparseLinearFunction.forward output\n", output[::stride, ::stride])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        check = False
        verbose = False
        input, weight_data = ctx.saved_tensors
        weight = ctx.weight
        assert isinstance(weight, BlockSparseMatrixBase)

        if verbose or check:
            dense_weight = weight.to_dense()

        if verbose:
            stride = 8
            print("input\n", input[::stride, ::stride])
            print(
                "grad_output\n",
                grad_output.stride(),
                grad_output.storage,
                grad_output.layout,
                grad_output[::stride, ::stride],
            )
            print("dense_weight\n", dense_weight[::stride, ::stride])
            print("weight\n", weight.data[::stride, ::stride])

        if ctx.needs_input_grad[0]:
            grad_input1 = weight.reverse_matmul(grad_output, transpose=False)

            if verbose or check:
                grad_input0 = grad_output.matmul(dense_weight)
                atol = 1e-4

                if check:
                    if not grad_input0.isclose(grad_input1).all():
                        print(f"grad_output.shape={grad_output.shape}, grad_output.stride={grad_output.stride()}")
                        print(
                            "grad_input0/1 comparison\n",
                            (grad_input0 - grad_input1)[1::32, 1::32, 1::32],
                        )
                        print(
                            "grad_input0/1 comparison\n",
                            (grad_input0 - grad_input1).abs().max(),
                        )
                        print(
                            "grad_input0/1 comparison: count of differences\n",
                            ((grad_input0 - grad_input1).abs() > atol).sum(),
                        )
                        print(
                            "grad_input0/1 comparison: position of differences\n",
                            ((grad_input0 - grad_input1).abs() > atol).nonzero(as_tuple=False),
                        )

                        print("grad_input0 max\n", grad_input0.abs().max())
                        print("grad_input1 max\n", grad_input1.abs().max())

                        raise Exception("Non matching grad_input")
                    else:
                        if verbose:
                            print("Backward matching grad_input")

                if verbose:
                    grad_input2 = weight.reverse_matmul(torch.ones_like(grad_output), transpose=False)
                    print("grad_input0\n", grad_input0[::stride, ::stride])
                    print("grad_input1\n", grad_input1[::stride, ::stride])
                    print("grad_input2\n", grad_input2[::stride, ::stride])
        else:
            grad_input1 = None

        if ctx.needs_input_grad[1]:
            grad_weight1 = weight.matmul_with_output_sparse_support(grad_output, input)
            if verbose or check:
                grad_weight0 = (
                    grad_output.reshape(-1, grad_output.shape[-1])
                    .transpose(-1, -2)
                    .matmul(input.reshape(-1, input.shape[-1]))
                )
                if check:
                    grad_weight1b = weight.to_dense(data_replace=grad_weight1)
                    grad_weight1mask = weight.to_dense(data_replace=torch.ones_like(grad_weight1))
                    grad_weight0 *= grad_weight1mask

                    if not grad_weight0.isclose(grad_weight1b).all():
                        print("grad_weight0\n", grad_weight0[::stride, ::stride])
                        print("grad_weight1\n", grad_weight1[::stride, ::stride])
                        raise Exception("Non matching grad_weight")
                    else:
                        if verbose:
                            print("Backward matching grad_weight")

                if verbose:
                    print("grad_weight0\n", grad_weight0[::stride, ::stride])
                    print("grad_weight1\n", grad_weight1[::stride, ::stride])
        else:
            grad_weight1 = None

        if grad_weight1 is not None:
            assert not (grad_weight1 == 0).all()
        if grad_input1 is not None:
            assert grad_input1.shape == input.shape

        return grad_input1, grad_weight1, None


class BlockSparseLinear(nn.Module):
    OPTIMIZED_BLOCK_SIZE = 32

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        density: float = 0.5,
        torch_nn_linear=None,
        verbose: bool = False,
        block_shape: Tuple[int, int] = (32, 32),
    ):
        super(BlockSparseLinear, self).__init__()
        self.fn = BlockSparseLinearFunction.apply
        self.verbose = verbose
        self.block_shape = block_shape
        self._optimized = (
            self.block_shape[0] == self.OPTIMIZED_BLOCK_SIZE and self.block_shape[1] == self.OPTIMIZED_BLOCK_SIZE
        )

        if torch_nn_linear is not None:
            in_features = torch_nn_linear.in_features
            out_features = torch_nn_linear.out_features
            bias = torch_nn_linear.bias is not None

        if in_features % self.block_shape[1] != 0:
            raise Exception(
                f"BlockSparseLinear invalid in_features={in_features}, should be multiple of {self.block_shape[1]}"
            )
        if out_features % self.block_shape[0] != 0:
            raise Exception(
                f"BlockSparseLinear invalid in_features={in_features}, should be multiple of {self.block_shape[0]}"
            )

        if density is None:
            block_count = None
        elif density < 0 or density > 1:
            raise Exception(f"BlockSparseLinear invalid density={density}")
        else:
            block_count = int(density * (in_features * out_features / (self.block_shape[0] * self.block_shape[1])))

        self.in_features = in_features
        self.out_features = out_features

        block_shape = self.block_shape

        if self._optimized:
            BlockSparseMatrixConstructor = BlockSparseMatrix
        else:
            BlockSparseMatrixConstructor = BlockSparseMatrixEmulator

        exact = block_count is None

        if torch_nn_linear is not None:
            with torch.no_grad():
                weight = BlockSparseMatrixConstructor.from_dense(
                    torch_nn_linear.weight, block_shape, block_count, exact=exact
                )
            block_count = weight.nnz_block_count()
            if density != None:
                weight.multiply_(1.0 / math.sqrt(density))
        else:
            weight = BlockSparseMatrixConstructor.randn(
                (out_features, in_features),
                block_count,
                blocks=None,
                block_shape=block_shape,
                device="cuda",
            )
        self.weight = weight

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device="cuda"))
            if torch_nn_linear is not None:
                with torch.no_grad():
                    self.bias.copy_(torch_nn_linear.bias)
        else:
            self.register_parameter("bias", None)

        self.block_count = block_count
        weight_blocks_count = self.weight.blocks_count()
        print("Final density:", block_count / weight_blocks_count[0] / weight_blocks_count[1])

    def forward(self, x):
        x = self.fn(x, self.weight.get_differentiable_data(), self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x


class PseudoBlockSparseLinear(torch.nn.Module):
    """For debugging purposes mostly: emulate a BlockSparseLinear with only PyTorch primitives."""

    def __init__(self, block_sparse_linear):
        super(PseudoBlockSparseLinear, self).__init__()

        block_sparse_matrix = block_sparse_linear.weight.cuda()
        self.weight = torch.nn.Parameter(block_sparse_matrix.to_dense())
        mask = block_sparse_matrix.to_dense(data_replace=torch.ones_like(block_sparse_matrix.data)) == 1
        if block_sparse_linear.bias is not None:
            self.bias = torch.nn.Parameter(block_sparse_linear.bias)
        else:
            self.register_parameter("bias", None)

        self.register_buffer("mask", mask)
        self.in_features = block_sparse_linear.in_features
        self.out_features = block_sparse_linear.out_features
        self.density = mask.sum().item() / (mask.shape[0] * mask.shape[1])

    def forward(self, input):
        weight = self.weight * self.mask
        return torch.nn.functional.linear(input, weight, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, fill_ratio={}".format(
            self.in_features, self.out_features, self.bias is not None, self.density
        )
