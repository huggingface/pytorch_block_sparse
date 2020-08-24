import torch
import torch.autograd
import torch.nn as nn
from .block_sparse import BlockSparseMatrix

class BlockSparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_data, weight):
        verbose = False
        if verbose:
            print("FORWARD STREAM", torch.cuda.current_stream())
        assert(isinstance(weight, BlockSparseMatrix))
        ctx.save_for_backward(input)
        ctx.weight = weight
        output = weight.reverse_matmul(input, transpose = True)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        verbose = False
        if verbose:
            print("BACKWARD STREAM", torch.cuda.current_stream())
        input, = ctx.saved_tensors
        weight = ctx.weight
        assert (isinstance(weight, BlockSparseMatrix))
        if False:
            dense_weight = weight.to_dense()
            stride = 8
            print("input\n", input[::stride, ::stride])
            print("grad_output\n", grad_output.stride(), grad_output.storage, grad_output.layout, grad_output[::stride, ::stride])
            print("dense_weight\n", dense_weight[::stride, ::stride])
            print("weight\n", weight.data[::stride, ::stride])

        if ctx.needs_input_grad[0]:
            grad_input1 = weight.reverse_matmul(grad_output, transpose=False)

            if False:
                grad_input0 = grad_output.mm(dense_weight)
                grad_input2 = weight.reverse_matmul(torch.ones_like(grad_output), transpose=False)

                print("grad_input0\n", grad_input0[::stride,::stride])
                print("grad_input1\n", grad_input1[::stride, ::stride])
                print("grad_input2\n", grad_input2[::stride, ::stride])

        if ctx.needs_input_grad[1]:
            grad_weight1 = weight.matmul_with_output_sparse_support(grad_output, input)
            if False:
                grad_weight0 = grad_output.t().mm(input)
                print("grad_weight0\n", grad_weight0[::stride, ::stride])
                print("grad_weight1\n", grad_weight1[::stride, ::stride])

        return grad_input1, grad_weight1, None

class BlockSparseLinear(nn.Module):
    BLOCK_SIZE=32
    def __init__(self,  in_features: int, out_features: int, bias: bool = True, density:float = 0.5):
        super(BlockSparseLinear, self).__init__()
        #self.fn = BlockSparseLinearFunction.apply

        if in_features % self.BLOCK_SIZE != 0:
            raise Exception(f"BlockSparseLinear invalid in_features={in_features}, should be multiple of {self.BLOCK_SIZE}")
        if out_features % self.BLOCK_SIZE != 0:
            raise Exception(f"BlockSparseLinear invalid in_features={in_features}, should be multiple of {self.BLOCK_SIZE}")

        if density < 0 or density > 1:
            raise Exception(f"BlockSparseLinear invalid density={density}")

        block_count = int(density * (in_features * out_features / (self.BLOCK_SIZE * self.BLOCK_SIZE)))

        self.in_features = in_features
        self.out_features = out_features

        weight = BlockSparseMatrix.randn((out_features, in_features),
                                         block_count,
                                         blocks=None,
                                         block_shape=(self.BLOCK_SIZE, self.BLOCK_SIZE),
                                         device="cuda")
        self.weight = weight

        self.weight_data = torch.nn.Parameter(self.weight.data)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        x = BlockSparseLinearFunction.apply(x, self.weight_data, self.weight)
        #if self.bias is not None:
        #    x += self.bias
        return x

