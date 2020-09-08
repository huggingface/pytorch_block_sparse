import torch


class SparseOptimizerStrategy:
    def run(self, block_sparse_matrix):
        raise NotImplementedError()


class MagnitudeSparseOptimizerStrategy(SparseOptimizerStrategy):
    def __init__(self, ratio, new_coefficients_method = "discrete", new_coefficients_scale=0.1):
        self.ratio = ratio
        self.new_coefficients_method = new_coefficients_method
        self.new_coefficients_scale = new_coefficients_scale

    def initialize_new_blocks(self, old_data, new_data):
        mean, std = old_data.mean(), old_data.std()

        if self.new_coefficients_method == "gaussian":
            new_data.normal_(mean=mean * self.new_coefficients_scale, std=std * self.new_coefficients_scale)
        elif self.new_coefficients_method == "discrete":
            new_data.random_(0, 1)
            new_data -= 0.5
            new_data *= 2 * std * self.new_coefficients_scale
        else:
            raise Exception("Unknown new coefficients method %s" % self.new_coefficients_method)

    def run(self, block_sparse_matrix):
        bsm = block_sparse_matrix
        # Get the norm of each block
        norms = bsm.block_norm()

        # Sort the norm
        _, indices = norms.sort()

        # Extract the worst blocks
        bad_blocks = indices[:int(indices.shape[0] * self.ratio)]

        # Find available positions
        block_mask = ~ bsm.block_mask_build(None)
        available = block_mask.nonzero()

        # Extract some random position
        empty_positions_indices = torch.randperm(available.shape[0])[:bad_blocks.shape[0]]
        new_positions = available[empty_positions_indices]

        block_replacements = torch.cat([new_positions, bad_blocks.unsqueeze(-1)], -1)

        bsm.block_replace(block_replacements)

        # bad_blocks
        new_block_mask = torch.zeros(bsm.data.shape[0] // bsm.block_shape[0], dtype=torch.bool, device = bsm.data.device)

        new_block_mask[bad_blocks] = True

        new_block_mask = new_block_mask.unsqueeze(-1).repeat(bsm.block_shape).float()

        new_blocks = torch.zeros_like(bsm.data)

        self.initialize_new_blocks(bsm.data, new_blocks)

        new_blocks *= new_block_mask

        with torch.no_grad():
            bsm.data *= 1.0 - new_block_mask
            bsm.data += new_blocks





