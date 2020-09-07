import unittest
from unittest import TestCase
import torch
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from pytorch_block_sparse import BlockSparseModelPatcher



class TestFun(TestCase):

    def helper(self, model, input_tensor, patterns, patch_info, param_counts):
        for i in range(2):
            parameter_count = 0
            for param in model.parameters():
                parameter_count += param.numel()

            self.assertEqual(parameter_count, param_counts[i])

            if i == 0:
                mp = BlockSparseModelPatcher()
                for p in patterns:
                    mp.add_pattern(p, patch_info)
                mp.patch_model(model)
            out = model(input_tensor)

    def test1(self):
        density = 0.5
        for bias in [False, True]:
            for patch_info in [{"density":0.5}, {"density":density, "pseudo_linear":True}]:
                linear = torch.nn.Linear(64, 128, bias)
                model = torch.nn.Sequential(linear).cuda()
                input_tensor = torch.randn(64, 64).cuda()

                pc = linear.weight.numel()
                if "pseudo_linear" in patch_info:
                    pc_sparse = pc
                else:
                    pc_sparse = int(pc * density)

                if bias:
                    pc += linear.bias.numel()
                    pc_sparse += linear.bias.numel()

                self.helper(model, input_tensor, ["0"], patch_info=patch_info, param_counts=[pc, pc_sparse])


    def test0(self):
        config = RobertaConfig(
            vocab_size=52_000,
            max_position_embeddings=514,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )

        model = RobertaForMaskedLM(config=config).cuda()
        model.eval()

        verbose = False

        for i in range(2):
            # => 70 million parameters instead of 84 million parameters when i = 1
            print("model num parameters", model.num_parameters())

            input_ids = torch.tensor([[4, 5, 6, 7]*8]).cuda()
            input_ids = input_ids.expand((1, 32))
            out = model(input_ids)
            if verbose:
                print(out)
            if i == 0:
                mp = BlockSparseModelPatcher()
                mp.add_pattern("roberta\.encoder\.layer\.[0-9]+.intermediate\.dense", {"density":0.5})
                mp.add_pattern("roberta\.encoder\.layer\.[0-9]+.output\.dense", {"density":0.5})
                mp.patch_model(model)

if __name__ == '__main__':
    unittest.main()
