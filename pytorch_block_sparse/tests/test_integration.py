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

    def test0(self):
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

    def roberta_build(self, sparse = False, base_model=None, density = 1.0, eval = True):
        if base_model == None:
            config = RobertaConfig(
                vocab_size=52_000,
                max_position_embeddings=514,
                num_attention_heads=12,
                num_hidden_layers=6,
                type_vocab_size=1,
            )

            model = RobertaForMaskedLM(config=config).cuda()
        else:
            model = base_model

        if sparse:
            mp = BlockSparseModelPatcher()
            mp.add_pattern("roberta\.encoder\.layer\.[0-9]+.intermediate\.dense", {"density": density})
            mp.add_pattern("roberta\.encoder\.layer\.[0-9]+.output\.dense", {"density": density})
            mp.patch_model(model)

        if eval:
            model.eval()

        return model, model.num_parameters()


    def test1(self):
        model0, num_parameters0 = self.roberta_build()

        input_ids = torch.tensor([[4, 5, 6, 7] * 8]).cuda()
        input_ids = input_ids.expand((1, 32))

        out0 = model0(input_ids)

        model1, num_parameters1 = self.roberta_build(sparse = True, base_model=model0)
        out1 = model1(input_ids)

        self.assertTrue(torch.isclose(out0[0], out1[0], atol=1e-3).all())

        model2, num_parameters2 = self.roberta_build(sparse=True, density = 0.5, eval=True)
        model2.eval()

        out2 = model2(input_ids)

        self.assertEqual(num_parameters0, num_parameters1)
        self.assertGreater(70000000, num_parameters2)

    def test_full(self):
            pass


if __name__ == '__main__':
    unittest.main()
