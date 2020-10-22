import os
import pathlib
import tempfile
import unittest
from typing import Any, Dict, Union
from unittest import TestCase

import torch
import torch.nn as nn
from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)

from pytorch_block_sparse import BlockSparseModelPatcher, SparseOptimizer


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
            _ = model(input_tensor)

    def test0(self):
        density = 0.5
        for bias in [False, True]:
            for patch_info in [
                {"density": 0.5},
                {"density": density, "pseudo_linear": True},
            ]:
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

                self.helper(
                    model,
                    input_tensor,
                    ["0"],
                    patch_info=patch_info,
                    param_counts=[pc, pc_sparse],
                )

    def roberta_build(self, sparse=False, base_model=None, density=1.0, eval=True):
        if base_model is None:
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
            mp.add_pattern(
                "roberta\\.encoder\\.layer\\.[0-9]+.intermediate\\.dense",
                {"density": density},
            )
            mp.add_pattern("roberta\\.encoder\\.layer\\.[0-9]+.output\\.dense", {"density": density})
            mp.patch_model(model)

        if eval:
            model.eval()

        return model, model.num_parameters()

    def test1(self):
        model0, num_parameters0 = self.roberta_build()

        input_ids = torch.tensor([[4, 5, 6, 7] * 8]).cuda()
        input_ids = input_ids.expand((1, 32))

        out0 = model0(input_ids)

        model1, num_parameters1 = self.roberta_build(sparse=True, base_model=model0)
        out1 = model1(input_ids)

        self.assertTrue(torch.isclose(out0[0], out1[0], atol=1e-3).all())

        model2, num_parameters2 = self.roberta_build(sparse=True, density=0.5, eval=True)
        model2.eval()

        _ = model2(input_ids)

        self.assertEqual(num_parameters0, num_parameters1)
        self.assertGreater(70000000, num_parameters2)

    def test_with_trainer(self):
        test_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
        data_dir = test_dir / "data"

        with tempfile.TemporaryDirectory() as tmpdir:
            model, num_parameters = self.roberta_build(sparse=True, density=0.5, eval=False)

            tokenizer = RobertaTokenizerFast.from_pretrained(str(data_dir), max_len=512)

            from transformers import DataCollatorForLanguageModeling

            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

            from transformers import LineByLineTextDataset

            dataset = LineByLineTextDataset(
                tokenizer=tokenizer,
                file_path=data_dir / "oscar.eo.small.txt",
                block_size=128,
            )

            training_args = TrainingArguments(
                output_dir=tmpdir,
                num_train_epochs=1,
                per_device_train_batch_size=16,  # Adapt it to your size
                save_steps=10_000,
            )

            class CustomTrainer(Trainer):
                def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
                    if self.first_step:
                        so.attach_optimizer(self.optimizer)
                    self.first_step = False
                    self.sparse_optimizer.step()
                    ret = super().training_step(model, inputs)
                    return ret

            trainer = CustomTrainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset,
            )

            cleanup_ratio = 0.1
            sparse_objects = SparseOptimizer.sparse_objects(model)

            self.assertEqual(len(sparse_objects), 12)
            so = SparseOptimizer(sparse_objects, lr=cleanup_ratio)

            trainer.sparse_optimizer = so
            trainer.first_step = True

            trainer.train()


if __name__ == "__main__":
    unittest.main()
