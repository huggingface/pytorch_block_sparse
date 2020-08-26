import re
import torch
from pytorch_block_sparse.block_sparse_linear import BlockSparseLinear

class ModelPatcher():
    def __init__(self):
        self.patterns = []

    def add_pattern(self, pattern, pattern_info):
        self.patterns.append(dict(pattern=pattern, pattern_info=pattern_info))

    def pattern_match(self, module_name):
        for pattern_def in self.patterns:
            if re.match(pattern_def["pattern"], module_name):
                return True, pattern_def["pattern_info"]
        return False, -1

    def new_child_module(self, child_module_name, child_module, pattern_info):
        print(child_module_name)
        return None

    def replace_module(self, father, child_module_name, child_name, child_module, pattern_info):
        new_child_module = self.new_child_module(child_module_name, child_module, pattern_info)
        if new_child_module != None:
            setattr(father, child_name, new_child_module)

    def patch_model(self, model):
        modules = {}
        for k, v in model.named_modules():
            modules[k] = v
            match, pattern_info = self.pattern_match(k)
            if match:
                parts = k.split(".")
                father_module_name = ".".join(parts[:-1])
                child_name = parts[-1]
                father = modules[father_module_name]
                self.replace_module(father, k, child_name, v, pattern_info)

class SparseModelPatcher(ModelPatcher):
    def new_child_module(self, child_module_name, child_module, pattern_info):
        density = pattern_info["density"]
        print(f"Patching {child_module_name} with density={density}")
        if isinstance(child_module, torch.nn.Linear):
            print(child_module_name, child_module.in_features, child_module.out_features)
        return BlockSparseLinear(0, 0, False, torch_nn_linear = child_module, density = density, device="cuda")