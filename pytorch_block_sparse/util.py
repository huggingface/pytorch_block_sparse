import re
import torch
from pytorch_block_sparse.block_sparse_linear import BlockSparseLinear

class ModelPatcher():
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
                ret.append({"regexp":r, "layer":v})
        return ret

    def add_pattern(self, pattern, pattern_info):
        self.patterns.append(dict(pattern=pattern, pattern_info=pattern_info))

    def pattern_match(self, module_name):
        for pattern_def in self.patterns:
            if re.match(pattern_def["pattern"], module_name):
                return True, pattern_def["pattern_info"]
        return False, -1

    def new_child_module(self, child_module_name, child_module, pattern_info):
        raise NotImplementedError("Implement this in subclasses")

    def replace_module(self, father, child_module_name, child_name, child_module, pattern_info):
        new_child_module = self.new_child_module(child_module_name, child_module, pattern_info)
        if new_child_module != None:
            setattr(father, child_name, new_child_module)

    def patch_model(self, model):
        modules = {}
        for k, v in model.named_modules():
            modules[k] = v
            match, pattern_info = self.pattern_match(k)
            if match and self.is_patchable(k, v, raiseError=True):
                parts = k.split(".")
                father_module_name = ".".join(parts[:-1])
                child_name = parts[-1]
                father = modules[father_module_name]
                self.replace_module(father, k, child_name, v, pattern_info)

class BlockSparseModelPatcher(ModelPatcher):
    def is_patchable(self, module_name, module, raiseError):
        if isinstance(module, torch.nn.Linear):
            return True
        else:
            if raiseError:
                raise Exception(f"Cannot patch {module_name}: this is not a Linear layer:\n{module}")
            return False

    def new_child_module(self, child_module_name, child_module, pattern_info):
        density = pattern_info["density"]
        self.is_patchable(child_module_name, child_module, raiseError=True)
        print(f"Patching '{child_module_name}' with density={density}, in={child_module.in_features},"
              f" out={child_module.out_features},bias={child_module.bias is not None} ")
        return BlockSparseLinear(0, 0, False, torch_nn_linear = child_module, density = density)
