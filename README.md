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

## Future work
- Upgrade to latest CUTLASS version, to optimize speed for latest architectures (using Tensor Cores for example)
- Use the new Ampere 50% sparse pattern within blocks themselves: more information on the [Hugging Face Blog](https://medium.com/huggingface/sparse-neural-networks-2-n-gpu-performance-b8bc9ce950fc). 

## Installation
In the root directory just execute: 
```
python setup.py install 
```
