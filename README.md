# Block Sparse Matrices for Pytorch v0.1

This PyTorch extension provides a **drop-in replacement** for torch.nn.Linear using **block sparse matrices** instead of dense ones.
This allows very easy experimentation, as you just have to replace the Linear layers in your model by a sparse one.

## Motivation
The incentive to create this library is to let people test the idea that **sparse matrices can be used in neural networks**, instead of dense ones, without significantly altering the precision.  
 
This would be great news as sparse matrices allows savings in both space and compute: a **50% sparse matrix** will use **only 50% memory**, and theoretically will use only 50% of computation.
However, due to the very optimized nature of cuBLAS based torch.nn.Linear, this lib is slower, by roughly a factor of 2 (this may be improved in the future).
But the performance gain of using sparse matrices grows with the sparsity, so a **75% sparse matrix** is roughly **2x** faster than the dense equivalent.

This could prove useful, and could be combined with other methods like distillation and quantization to reduce further the networks.  

## Base code
This work is based on the [cutlass tilesparse](https://github.com/YulhwaKim/cutlass_tilesparse) proof of concept by [Yulhwa Kim](https://github.com/YulhwaKim).

It is using C++ CUDA templates for block-sparse matrix multiplication based on [CUTLASS](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/).

## Related work
OpenAI announced in January 2020 that their very advanced (and complex) TensorFlow code [would be ported](https://openai.com/blog/openai-pytorch/) to PyTorch.
Unfortunately this has not happened yet.

Google and Stanford June 2020 paper [Sparse GPU Kernels for Deep Learning](https://arxiv.org/abs/2006.10901) is promising too, as the code should be released at some time.
This would be even more general, as the sparsity pattern is not constrained, and the performance looks very good, with some smart ad hoc optimizations.

## Basic usage
You can use the BlockSparseLinear drop in replacement for torch.nn.Linear in your own model.

Or you can use a utility called BlockSparseModelPatcher to modify easily an existing model before training it.
(you cannot magically sparsify a trained existing model, you will need to train it from scratch)
Here is an example with a Roberta Model from Hugging Face ([full example](docs/notebooks/ModelSparsification.ipynb))

```python
from pytorch_block_sparse import BlockSparseModelPatcher
# Create a model patcher
mp = BlockSparseModelPatcher()

# Selecting some layers to sparsify.
# This is the "artful" part, as some parts are more prone to be sparsified, other may impact model precision too much.

# Match layers using regexp (we escape the ., just because, it's more correct, but it does not change anything here)
# the [0-9]+ match any layer number.
# We setup a density of 0.5 on these layers, you can test other layers / densities .
mp.add_pattern("roberta\.encoder\.layer\.[0-9]+\.intermediate\.dense", {"density":0.5})
mp.add_pattern("roberta\.encoder\.layer\.[0-9]+\.output\.dense", {"density":0.5})
mp.add_pattern("roberta\.encoder\.layer\.[0-9]+\.attention\.output\.dense", {"density":0.5})
mp.patch_model(model)

print(f"Final model parameters count={model.num_parameters()}")

# => 68 million parameters instead of 84 million parameters (embeddings are taking a lof space in Roberta)
```

## Future work
- Implement some paper methods (and provide new ones) to optimize the sparse pattern during training, while doing the classic parameter optimization using backprop. The base example is to remove some smaller magnitude weights (or blocks of weights) at some positions and try other ones.  
- Upgrade to latest CUTLASS version, to optimize speed for latest architectures (using Tensor Cores for example)
- Use the new Ampere 50% sparse pattern within blocks themselves: more information on the [Hugging Face Blog](https://medium.com/huggingface/sparse-neural-networks-2-n-gpu-performance-b8bc9ce950fc).

## Installation
In the root directory just execute: 
```
python setup.py install 
```
