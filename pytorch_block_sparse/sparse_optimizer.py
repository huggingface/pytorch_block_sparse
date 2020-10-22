import torch
import torch.optim as optim

from pytorch_block_sparse import BlockSparseMatrix

try:
    import transformers.optimization as transformers_optim
except Exception:
    transformers_optim = None


class SparseOptimizerStrategy:
    def run(self, block_sparse_matrix):
        raise NotImplementedError()


class MagnitudeSparseOptimizerStrategy(SparseOptimizerStrategy):
    def __init__(
        self,
        cleanup_ratio,
        new_coefficients_distribution="uniform",
        new_coefficients_scale=0.1,
    ):
        self.cleanup_ratio = cleanup_ratio
        self.new_coefficients_distribution = new_coefficients_distribution
        self.new_coefficients_scale = new_coefficients_scale

    def initialize_new_blocks(self, old_data, new_data):
        mean, std = old_data.mean(), old_data.std()

        if self.new_coefficients_distribution == "gaussian":
            new_data.normal_(
                mean=mean * self.new_coefficients_scale,
                std=std * self.new_coefficients_scale,
            )
        elif self.new_coefficients_distribution == "uniform":
            new_data.random_(0, 1)
            new_data -= 0.5
            new_data *= 2 * std * self.new_coefficients_scale
        else:
            raise Exception("Unknown new coefficients method %s" % self.new_coefficients_distribution)

    def run(self, block_sparse_matrix):
        bsm = block_sparse_matrix
        # Get the norm of each block
        norms = bsm.block_norm()

        # Sort the norm
        _, indices = norms.sort()

        # Extract the worst blocks
        bad_blocks = indices[: int(indices.shape[0] * self.cleanup_ratio)]

        # Find available positions
        block_mask = ~bsm.block_mask_build(None)
        available = block_mask.nonzero()

        # Extract some random position
        empty_positions_indices = torch.randperm(available.shape[0])[: bad_blocks.shape[0]]
        new_positions = available[empty_positions_indices]

        block_replacements = torch.cat([new_positions, bad_blocks.unsqueeze(-1)], -1)

        bsm.block_replace(block_replacements)

        # bad_blocks
        new_block_mask = torch.zeros(
            bsm.data.shape[0] // bsm.block_shape[0],
            dtype=torch.bool,
            device=bsm.data.device,
        )

        new_block_mask[bad_blocks] = True

        new_block_mask = new_block_mask.unsqueeze(-1)
        new_block_mask = new_block_mask.repeat_interleave(bsm.block_shape[0], dim=0)
        new_block_mask = new_block_mask.repeat_interleave(bsm.block_shape[1], dim=1)
        new_block_mask = new_block_mask.float()

        new_blocks = torch.zeros_like(bsm.data)

        self.initialize_new_blocks(bsm.data, new_blocks)

        new_blocks *= new_block_mask

        state_keep_mask = 1.0 - new_block_mask

        with torch.no_grad():
            bsm.data *= state_keep_mask
            bsm.data += new_blocks

        return state_keep_mask


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class OptimizerStateUpdater:
    def __init__(self, optimizer, sparse_object):
        self.optimizer = optimizer
        if not isinstance(sparse_object, BlockSparseMatrix):
            raise Exception(f"Unknown sparse_object type {sparse_object}")

        self.sparse_object = sparse_object

    def update_state_data(self, param, state_keep_mask):
        raise NotImplementedError()

    def update_state(self, state_keep_mask):
        if isinstance(self.sparse_object, BlockSparseMatrix):
            search_param = self.sparse_object.data
        else:
            raise Exception(f"Unknown sparse_object type {self.sparse_object}")

        found = False
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                if param is search_param:
                    found = True
                    self.update_state_data(param, state_keep_mask)

        return found


class AdamOptimizerStateUpdater(OptimizerStateUpdater):
    @staticmethod
    def is_compatible(optimizer):
        if isinstance(optimizer, optim.Adam):
            return True

        if transformers_optim is not None:
            if isinstance(optimizer, transformers_optim.AdamW):
                return True

    def update_state_data(self, param, state_keep_mask):
        opt = self.optimizer

        param_state = opt.state[param]

        for key in param_state:
            if key in ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]:
                param_state[key] *= state_keep_mask
            elif key == "step":
                # We cannot really alter the step info, it's global, so the bias_correction1 and bias_correction2 may
                # not be completely correct for the new coefficients, but it should not be a big issue
                pass
            else:
                raise Exception(f"Unknown key in Adam parameter state {key}")


class SparseOptimizer(torch.optim.Optimizer):
    METHODS = ["magnitude"]
    COEFFICIENTS_DISTRIBUTION = ["uniform", "gaussian"]
    allowed_keys = {
        "lr",
        "method",
        "new_coefficients_scale",
        "new_coefficients_distribution",
    }
    """optimizer = sparse_cleaner.SparseOptimizer([BlockSparseMatrix,BlockSparseMatrix],
                                                  method="magnitude", new_coefficients_distribution="uniform")
       optimizer.add_param_group(dict(sparse_objects=[BlockSparseMatrix],
                                      lr=0.5,  method="magnitude",
                                       new_coefficients_distribution="gaussian", new_coefficients_scale = 1.0))"""

    def __init__(
        self,
        sparse_objects,
        lr=1e-1,
        method="magnitude",
        new_coefficients_scale=0.1,
        new_coefficients_distribution="uniform",
    ):
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(
            lr=lr,
            method=method,
            new_coefficients_scale=new_coefficients_scale,
            new_coefficients_distribution=new_coefficients_distribution,
        )

        super(SparseOptimizer, self).__init__([{"sparse_objects": sparse_objects}], defaults)
        self.attached_optimizers = []

    @staticmethod
    def sparse_objects(model):
        ret = []
        for name, module in model.named_modules():
            if isinstance(module, BlockSparseMatrix):
                ret.append(module)

        return ret

    def attach_optimizer(self, optimizer):
        if optimizer in self.attached_optimizers:
            Warning("Optimizer already attached")
            return
        self.attached_optimizers.append(optimizer)

    def add_param_group(self, sparse_objects_group):
        assert isinstance(sparse_objects_group, dict), "param group must be a dict"

        for k in sparse_objects_group:
            if k == "sparse_objects":
                continue
            elif k not in self.allowed_keys:
                raise Exception("Unknown cleaning parameter %s" % k)

        sparse_objects = sparse_objects_group["sparse_objects"]

        if isinstance(sparse_objects, BlockSparseMatrix):
            sparse_objects_group["sparse_objects"] = [sparse_objects]
        else:
            sparse_objects_group["sparse_objects"] = list(sparse_objects)

        sparse_objects = sparse_objects_group["sparse_objects"]

        for p in sparse_objects:
            if isinstance(p, BlockSparseMatrix):
                continue
            else:
                raise Exception("I don't know how to clean this type of object: %s" % p)

        for name, default in self.defaults.items():
            if default is required and name not in sparse_objects_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " + name)
            else:
                sparse_objects_group.setdefault(name, default)

        if sparse_objects_group["method"] not in self.METHODS:
            raise Exception(f"Invalid Method {sparse_objects_group['method']}")

        if sparse_objects_group["new_coefficients_distribution"] not in self.COEFFICIENTS_DISTRIBUTION:
            raise Exception(
                f"Invalid new coefficients distribution {sparse_objects_group['new_coefficients_distribution']}"
            )

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group["sparse_objects"]))

        if not param_set.isdisjoint(set(sparse_objects_group["sparse_objects"])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(sparse_objects_group)

    def clean(
        self,
        p,
        method,
        clean_ratio,
        new_coefficients_scale,
        new_coefficients_distribution,
    ):
        if not isinstance(p, BlockSparseMatrix):
            raise Exception("I don't know how to clean this : %s" % p)

        if method == "magnitude":
            cleaner = MagnitudeSparseOptimizerStrategy(
                clean_ratio,
                new_coefficients_distribution=new_coefficients_distribution,
                new_coefficients_scale=new_coefficients_scale,
            )
        else:
            raise Exception(f"Unknowncleaning method {method}")

        state_keep_mask = cleaner.run(p)

        if len(self.attached_optimizers) != 0:
            found = False
            for optimizer in self.attached_optimizers:
                if AdamOptimizerStateUpdater.is_compatible(optimizer):
                    updater = AdamOptimizerStateUpdater(optimizer, p)
                    found = found or updater.update_state(state_keep_mask)
                else:
                    raise Exception(f"unsupported optimizer {optimizer.__class__}")

            if not found:
                raise Exception(f"Could not find sparse object {p} in optimizers {self.attached_optimizers}")
        else:
            Warning("No attached optimizer.")

    def step(self):
        for group in self.param_groups:
            clean_ratio = group["lr"]
            if clean_ratio == 0.0:
                continue
            for p in group["sparse_objects"]:
                self.clean(
                    p,
                    clean_ratio=clean_ratio,
                    method=group["method"],
                    new_coefficients_scale=group["new_coefficients_scale"],
                    new_coefficients_distribution=group["new_coefficients_distribution"],
                )
