# Troubleshooting

## Locate the problem
If your model has an unexpected behaviour, and you suspect that pytorch_block_sparse maybe the culprit,
you can easily test the hypothesis.
- First make sure that your model works with a dense version, aka torch.nn.Linear.
 
- Then, if everything is ok, use the `block_sparse_linear.PseudoBlockSparseLinear` layer,
using your `BlockSparseLinear` object to initialize it, instead of directly using the `BlockSparseLinear`.
`PseudoBlockSparseLinear` use only PyTorch primitives, so if your model has still issues after that change,
that means that the CUDA kernels are not responsible and that
the problem may be that sparsity creates learning instability, or that there is a problem in you own code.

- If your issue disappear after using `PseudoBlockSparseLinear`,
please fill a PR with the details to reproduce it, we will be glad to investigate. 
  
## Helper included in BlockSparseModelPatcher

If you are using `BlockSparseModelPatcher`, there is an easy way to switch to PseudoBlockSparseLinear.
Just use the `"pseudo_linear":True` (key,value) in the `add_pattern(...,patch_info=)` parameter: 


```python
from pytorch_block_sparse import BlockSparseModelPatcher
# Create a model patcher
mp = BlockSparseModelPatcher()

# Selecting some layers to sparsify.
# We use a PseudoBlockSparseLayer to check CUDA kernels 
mp.add_pattern("roberta\.encoder\.layer\.[0-9]+\.intermediate\.dense", patch_info={"density":0.5, "pseudo_linear":True})
mp.add_pattern("roberta\.encoder\.layer\.[0-9]+\.output\.dense", patch_info={"density":0.5, "pseudo_linear":True})
mp.add_pattern("roberta\.encoder\.layer\.[0-9]+\.attention\.output\.dense", patch_info={"density":0.5, "pseudo_linear":True})
mp.patch_model(model)

print(f"Final model parameters count={model.num_parameters()}")

```
