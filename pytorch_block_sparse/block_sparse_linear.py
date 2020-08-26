import torch
import torch.autograd
import torch.nn as nn
from .block_sparse import BlockSparseMatrix
import typing

class BlockSparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_data, full_weight, weight):
        check = True
        verbose = False
        if verbose:
            print("FORWARD\n", input, weight_data, weight)
        assert(isinstance(weight, BlockSparseMatrix))
        ctx.save_for_backward(input, weight_data, full_weight)
        ctx.weight = weight
        output = weight.reverse_matmul(input, transpose = True)
        if check:
            if full_weight is not None:
                output2 = input.matmul(full_weight.t())
                if not output2.isclose(output, atol=1e-05).all():
                    raise Exception("FORWARD non matching output 0")
                else:
                    if verbose:
                        print("FORWARD matching output 0")

            dense = weight.to_dense()

            if full_weight is not None:
                if not dense.isclose(full_weight, atol=1e-05).all():
                    raise Exception("FORWARD non matching matrices")

            output3 = input.matmul(dense.t())
            if not output3.isclose(output, atol=1e-05).all():
                raise Exception("FORWARD non matching output 1")
            else:
                if verbose:
                    print("FORWARD matching output 1")
        if verbose:
            print("FORWARD output\n", output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        check = True
        verbose = False
        if verbose:
            print("BACKWARD STREAM", torch.cuda.current_stream())
        input,weight_data, full_weight = ctx.saved_tensors
        weight = ctx.weight
        weight.data = weight_data
        assert (isinstance(weight, BlockSparseMatrix))
        stride = 8
        if verbose or check:
            dense_weight = weight.to_dense()
            if verbose:
                print("input\n", input[::stride, ::stride])
                print("grad_output\n", grad_output.stride(), grad_output.storage, grad_output.layout, grad_output[::stride, ::stride])
                print("dense_weight\n", dense_weight[::stride, ::stride])
                print("weight\n", weight.data[::stride, ::stride])
        #print(ctx.needs_input_grad)
        if ctx.needs_input_grad[0]:
            #grad_output = grad_output.contiguous()
            grad_input1 = weight.reverse_matmul(grad_output, transpose=False)

            if verbose or check:
                grad_input0 = grad_output.matmul(dense_weight)

                if check:
                    #atol = grad_output.abs().max() * 1e-02
                    atol = 0.0001
                    if verbose:
                        print(f"atol={atol}")
                    if not grad_input0.isclose(grad_input1, atol=atol).all():
                        print(f"grad_output.shape={grad_output.shape}, grad_output.stride={grad_output.stride()}")
                        print("grad_input0/1 comparison\n", (grad_input0 - grad_input1)[1::32,1::32,1::32])
                        print("grad_input0/1 comparison\n", (grad_input0 - grad_input1).abs().max())
                        print("grad_input0/1 comparison: count of differences\n", ((grad_input0 - grad_input1).abs() > atol).sum())
                        print("grad_input0/1 comparison: position of differences\n",
                              ((grad_input0 - grad_input1).abs() > atol).nonzero())

                        print("grad_input0 max\n", grad_input0.abs().max())
                        print("grad_input1 max\n", grad_input1.abs().max())

                        raise Exception("Non matching grad_input")
                    else:
                        if verbose:
                            print("Backward matching grad_input")

                if verbose:
                    grad_input2 = weight.reverse_matmul(torch.ones_like(grad_output), transpose=False)
                    print("grad_input0\n", grad_input0[::stride,::stride])
                    print("grad_input1\n", grad_input1[::stride, ::stride])
                    print("grad_input2\n", grad_input2[::stride, ::stride])
        else:
            grad_input1 = None

        if ctx.needs_input_grad[1]:
            grad_weight1 = weight.matmul_with_output_sparse_support(grad_output, input)
            if verbose or check:
                #print("shapes", grad_output.shape, input.shape)
                grad_weight0 = grad_output.reshape(-1, grad_output.shape[-1]).transpose(-1,-2).matmul(input.reshape(-1, input.shape[-1]))
                if check:
                    grad_weight1b = weight.to_dense(data_replace=grad_weight1)
                    #grad_weight1mask = weight.to_dense(data_replace=torch.ones_like(grad_weight1))
                    #grad_weight0 *= grad_weight1mask
                    if verbose:
                        print("grad_weight0.shape", grad_weight0.shape)
                        print("grad_weight1b.shape", grad_weight1b.shape)

                    if not grad_weight0.isclose(grad_weight1b, atol=1e-05).all():
                        print(grad_weight0[::32,::32])
                        print(grad_weight1b[::32, ::32])
                        raise Exception("Non matching grad_weight")
                    else:
                        if verbose:
                            print("Backward matching grad_weight")

                if verbose:
                    print("grad_weight0\n", grad_weight0[::stride, ::stride])
                    print("grad_weight1\n", grad_weight1[::stride, ::stride])
        else:
            grad_weight1 = None

        if ctx.needs_input_grad[0]:
            if full_weight is not None:
                grad_input_full = grad_output.matmul(full_weight)
                if check:
                    if not grad_input_full.isclose(grad_input1, atol=1e-03).all():
                        print(grad_input_full[::32, ::32])
                        print(grad_input1[::32, ::32])
                        raise Exception("Backward non matching full grad_input")
                    else:
                        if verbose:
                            print("Backward matching full grad_input")
            else:
                grad_input_full = None

        if ctx.needs_input_grad[1]:
            if full_weight is not None:
                grad_weight_full = grad_output.reshape(-1, grad_output.shape[-1]).transpose(-1, -2).matmul(input.reshape(-1, input.shape[-1]))
                if check:
                    if not grad_weight_full.isclose(grad_weight1b, atol=1e-03).all():
                        print(grad_weight_full[::32, ::32])
                        print(grad_weight1b[::32, ::32])
                        raise Exception("Backward non matching full grad_weight")
                    else:
                        if verbose:
                            print("Backward matching full grad_weight")
            else:
                grad_weight_full = None

        assert(not (grad_weight1 == 0).all())
        assert(grad_input0.shape == input.shape)
        return grad_input1, grad_weight1, grad_weight_full, None

class BlockSparseLinear(nn.Module):
    BLOCK_SIZE=32
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 density:float = 0.5,
                 torch_nn_linear = None,
                 verbose = False):
        super(BlockSparseLinear, self).__init__()
        self.fn = BlockSparseLinearFunction.apply
        self.verbose = verbose

        if torch_nn_linear != None:
            in_features = torch_nn_linear.in_features
            out_features = torch_nn_linear.out_features
            bias = torch_nn_linear.bias is not None

        if in_features % self.BLOCK_SIZE != 0:
            raise Exception(f"BlockSparseLinear invalid in_features={in_features}, should be multiple of {self.BLOCK_SIZE}")
        if out_features % self.BLOCK_SIZE != 0:
            raise Exception(f"BlockSparseLinear invalid in_features={in_features}, should be multiple of {self.BLOCK_SIZE}")

        if density < 0 or density > 1:
            raise Exception(f"BlockSparseLinear invalid density={density}")

        block_count = int(density * (in_features * out_features / (self.BLOCK_SIZE * self.BLOCK_SIZE)))

        self.in_features = in_features
        self.out_features = out_features

        block_shape = (self.BLOCK_SIZE, self.BLOCK_SIZE)
        if torch_nn_linear is not None:
            with torch.no_grad():
                weight = BlockSparseMatrix.from_dense(torch_nn_linear.weight, block_shape, density)
                if self.verbose:
                    print("weight data\n", weight.data)
                weight.check_with_dense(torch_nn_linear.weight)
        else:
            weight = BlockSparseMatrix.randn((out_features, in_features),
                                             block_count,
                                             blocks=None,
                                             block_shape=block_shape,
                                             device="cuda")
        self.weight = weight

        self.weight_data = torch.nn.Parameter(self.weight.data)
        #self.weight.data = None

        if torch_nn_linear is not None:
            self.full_weight = nn.Parameter(torch.zeros([out_features, in_features], device="cuda"))
            with torch.no_grad():
                self.full_weight[::, ::] = torch_nn_linear.weight
        else:
            self.register_parameter('full_weight', None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device = "cuda"))
            if torch_nn_linear is not None:
                with torch.no_grad():
                    self.bias.copy_(torch_nn_linear.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.verbose:
            print("x0=\n", x)
        #s = self.weight.data.isclose(self.weight_data).all()
        #assert(s)

        #print("weight_data[0,0]=", self.weight_data[0,0])
        #self.weight.data = None
        #self.weight.data = self.weight_data.data
        x = self.fn(x, self.weight_data, self.full_weight, self.weight)
        #self.weight.data = None
        if self.verbose:
            print("x1=\n", x)
        if self.bias is not None:
            #print("bias[0]", self.bias[0])
            x = x + self.bias
        return x


