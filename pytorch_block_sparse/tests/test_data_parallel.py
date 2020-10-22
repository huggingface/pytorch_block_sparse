import unittest
from unittest import TestCase

import torch
import torch.nn

from pytorch_block_sparse import BlockSparseLinear


class TestFun(TestCase):
    def test1(self):
        linear = BlockSparseLinear(64, 128, False).to("cuda")
        model = torch.nn.DataParallel(linear)

        input_tensor = torch.randn(64, 64).cuda()

        _ = model(input_tensor)


if __name__ == "__main__":
    unittest.main()
