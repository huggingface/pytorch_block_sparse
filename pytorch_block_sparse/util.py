import re

import torch

from pytorch_block_sparse import BlockSparseLinear
from pytorch_block_sparse.block_sparse_linear import PseudoBlockSparseLinear


class ModelPatcher:
    def __init__(self):
        self.patterns = []

    def is_patchable(self, module_name, module, raiseError):
        return True

    def get_patchable_layers(self, model):
        # Layer names (displayed as regexps)")
        ret = []
        for k, v in model.named_modules():
            if self.is_patchable(k, v, raiseError=False):
                r = re.escape(k)
                ret.append({"regexp": r, "layer": v})
        return ret

    def add_pattern(self, pattern, patch_info):
        self.patterns.append(dict(pattern=pattern, patch_info=patch_info))

    def pattern_match(self, module_name):
        for pattern_def in self.patterns:
            if re.match(pattern_def["pattern"], module_name):
                return True, pattern_def["patch_info"]
        return False, -1

    def new_child_module(self, child_module_name, child_module, patch_info):
        raise NotImplementedError("Implement this in subclasses")

    def replace_module(self, father, child_module_name, child_name, child_module, patch_info):
        new_child_module = self.new_child_module(child_module_name, child_module, patch_info)
        if new_child_module is not None:
            setattr(father, child_name, new_child_module)

    def patch_model(self, model):
        """Modes are block_sparse, or dense. Dense is just trying to remove entire columns/rows."""
        modules = {}
        modified = False
        for k, v in model.named_modules():
            modules[k] = v
            match, patch_info = self.pattern_match(k)
            if match and self.is_patchable(k, v, raiseError=True):
                parts = k.split(".")
                father_module_name = ".".join(parts[:-1])
                child_name = parts[-1]
                father = modules[father_module_name]
                self.replace_module(father, k, child_name, v, patch_info)
                modified = True
        if not modified:
            print(
                "Warning: the patcher did not patch anything!"
                " Check patchable layers with `mp.get_patchable_layers(model)`"
            )

class BertHeadsPruner():
    def __init__(self, model):
        self.model = model

    def analyze_head(self, p, head_size):
        p0 = (p != 0).reshape(p.shape[0] // head_size, head_size, p.shape[1]).any(-1).any(-1)
        return p0

    def get_pruned_heads(self):
        heads_count = 0
        to_prune = {}
        for name, module in self.model.named_modules():
            if name.endswith("attention.self"):
                layer_number = int(name.split(".")[3])
                parts = []
                for a in ["query", "key", "value"]:
                    p = self.analyze_head(getattr(module, a).weight, module.attention_head_size)
                    parts.append(p)
                parts = list(torch.stack(parts, 0).all(0).cpu().detach().numpy())
                heads_count += len(parts)

                heads_to_prune = [i for i, p in enumerate(parts) if not p]

                to_prune[layer_number] = heads_to_prune
        return to_prune, heads_count

    def run(self):
        model = self.model

        to_prune, heads_count = self.get_pruned_heads()

        model.prune_heads(to_prune)
        return sum([len(p) for p in to_prune.values()]), heads_count

class BlockSparseModelPatcher(ModelPatcher):
    """Use {"density":d} with d in [0,1] in patch_info}
    Use {"pseudo_linear":True} in patch_info to use a pytorch only implementation, if you think there is a bug
    in pytorch_block_sparse library"""

    def __init__(self, prune_heads=False, mode="block_sparse"):
        super().__init__()
        self.prune_heads = prune_heads
        self.mode = mode

    def is_patchable(self, module_name, module, raiseError):
        if isinstance(module, torch.nn.Linear):
            return True
        else:
            if raiseError:
                raise Exception(f"Cannot patch {module_name}: this is not a Linear layer:\n{module}")
            return False

    def new_child_module_block_sparse(self, child_module_name, child_module, patch_info):
        density = patch_info.get("density")
        pseudo = patch_info.get("pseudo_linear")
        if pseudo:
            patch_type = "PseudoBlockSparseLinear (debug)"
        else:
            patch_type = "BlockSparseLinear"

        self.is_patchable(child_module_name, child_module, raiseError=True)
        print(
            f"Patching with {patch_type} '{child_module_name}' with density={density}, in={child_module.in_features},"
            f" out={child_module.out_features},bias={child_module.bias is not None} "
        )
        ret = BlockSparseLinear(0, 0, False, torch_nn_linear=child_module, density=density)
        if pseudo:
            ret = PseudoBlockSparseLinear(ret)

        return ret

    def get_sparsity(self, w, dim):
        r = (w != 0).sum(dim)
        nnz = (r != 0).sum()
        return 1.0 - (nnz / r.numel()), r != 0


    def new_child_module_dense(self, child_module_name, child_module, patch_info):
        if "attention" in child_module_name:
            return None
        weight = child_module.weight
        device = weight.device
        bias = child_module.bias

        r_sparsity, r = self.get_sparsity(weight, 1)
        c_sparsity, c = self.get_sparsity(weight, 0)

        if r_sparsity > c_sparsity:
            weight = weight[r != 0]
            bias = bias[r != 0]
        else:
            weight = weight[:, c != 0]

        ret = torch.nn.Linear(weight.shape[1], weight.shape[0], bias = True).to(device)
        with torch.no_grad():
            ret.weight.copy_(weight)
            ret.bias.copy_(bias)
        return ret

    def new_child_module(self, child_module_name, child_module, patch_info):
        if self.mode == "block_sparse":
            return self.new_child_module_block_sparse(child_module_name, child_module, patch_info)
        elif self.mode == "dense":
            return self.new_child_module_dense(child_module_name, child_module, patch_info)

    def patch_model(self, model):
        if self.prune_heads:
            pruner = BertHeadsPruner(model)
            removed_heads, total_heads = pruner.run()
            print(f"removed heads {removed_heads}, total_heads={total_heads}, percentage removed={removed_heads/total_heads}")

        super().patch_model(model)
