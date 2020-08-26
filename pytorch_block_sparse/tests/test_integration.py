import unittest
from unittest import TestCase
import torch
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from pytorch_block_sparse.block_sparse_linear import BlockSparseLinear
from pytorch_block_sparse.util import SparseModelPatcher



class TestFun(TestCase):

    def helper(self, model, input_tensor, patterns):
        for i in range(2):
            print(f"i={i}\n", model(input_tensor))
            if i == 0:
                mp = SparseModelPatcher()
                for p in patterns:
                    mp.add_pattern(p)
                mp.patch_model(model)

    def tst1(self):
        linear = torch.nn.Linear(64, 128, False)
        model = torch.nn.Sequential(linear).cuda()
        input_tensor = torch.randn(64, 64).cuda()

        self.helper(model, input_tensor, ["0"])

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
            input_ids = torch.tensor([[4, 5, 6, 7]*8]).cuda()
            input_ids = input_ids.expand((1, 32))
            out = model(input_ids)
            if verbose:
                print(out)
            if i == 0:
                mp = SparseModelPatcher()
                mp.add_pattern("roberta\.encoder\.layer\.0.intermediate\.dense")
                mp.add_pattern("roberta\.encoder\.layer\..*\.output\.dense")
                mp.patch_model(model)

        # model.roberta.encoder.layer[5].intermediate.dense = torch.nn.Linear(768, 3072, True)

        # => 60 million parameters instead of 84 million parameters
        model.num_parameters()



if __name__ == '__main__':
    unittest.main()
