import unittest
from unittest import TestCase

import torch

from pytorch_block_sparse import BlockSparseMatrix


class TestFun(TestCase):
    def test_block_norm(self):
        nblocks = 6
        block_shape = (32, 32)
        bsm = BlockSparseMatrix.randn((256, 256), nblocks, block_shape=block_shape, device="cuda")
        n = bsm.block_norm()
        self.assertEqual(n.dim(), 1)
        self.assertEqual(n.shape[0], nblocks)

        d = bsm.data.reshape(-1, block_shape[0] * block_shape[1])
        d = (d * d).sum(-1).sqrt()
        self.assertTrue(d.isclose(n).all())

    def test_block_replace(self):
        tests = [
            dict(
                size=[128, 64],
                blocks=[
                    (0, 0),
                    (1, 0),
                    (2, 0),
                    (0, 1),
                ],
                block_info=[(0, 0), (0, 1), (1, 0), (2, 0)],
                block_replace=[
                    (3, 1, 0),
                    (2, 1, 2),
                    (1, 1, 1),
                ],  # row, col, block_index
                after=dict(
                    row_start_ends_a=[0, 0, 1, 3, 4],
                    cols_a=[[1, 1], [0, 3], [1, 2], [1, 0]],
                    block_mask=[[0, 0], [0, 1], [1, 1], [0, 1]],
                ),
            ),
            dict(
                size=[128, 64],
                blocks=[
                    (0, 0),
                    (1, 0),
                    (2, 0),
                    (0, 1),
                ],
                block_info=[(0, 0), (0, 1), (1, 0), (2, 0)],
                block_replace=[(0, 1, 0)],  # row, col, block_index
                error="Block position (0,1) was already used",
            ),
        ]
        block_shape = (32, 32)
        device = "cuda"
        verbose = False
        for test_info in tests[:1]:
            size = test_info["size"]
            blocks = test_info["blocks"]
            block_replace = torch.tensor(test_info["block_replace"])
            bsm = BlockSparseMatrix.randn(
                (size[0], size[1]),
                None,
                blocks=blocks,
                block_shape=block_shape,
                device=device,
                positive=True,
            )
            bsm.check_ = True

            if verbose:
                print(block_replace)
                block_mask0 = bsm.block_mask_build(None)
                print(block_mask0)

            dbsm0 = bsm.to_dense()
            block_positions = bsm.build_coo_block_index().t()
            for i, b in enumerate(test_info["block_info"]):
                block_position = tuple(block_positions[i].cpu().numpy())
                self.assertEqual(b, block_position)

            try:
                bsm.block_replace(block_replace)
            except Exception as e:
                if test_info.get("error") == str(e):
                    continue
                raise

            for k, v in test_info["after"].items():
                if k != "block_mask":
                    r = getattr(bsm, k)
                else:
                    r = bsm.block_mask_build(None).long()
                v = torch.tensor(v, device=r.device)

                self.assertTrue((r == v).all())

            dbsm = bsm.to_dense()
            bsm.check_with_dense(dbsm)

            # Check changed positions
            bs = block_shape
            for b in block_replace:
                block_index = b[2]
                bp = block_positions[block_index]
                block0 = dbsm0[
                    bp[0] * bs[0] : (bp[0] + 1) * bs[0],
                    bp[1] * bs[1] : (bp[1] + 1) * bs[1],
                ]
                block = dbsm[b[0] * bs[0] : (b[0] + 1) * bs[0], b[1] * bs[1] : (b[1] + 1) * bs[1]]

                self.assertTrue((block0 == block).all())

            # Check unchanged positions
            for i, b in enumerate(block_positions):
                if i not in block_replace[:, 2]:
                    bp = b
                    block0 = dbsm0[
                        bp[0] * bs[0] : (bp[0] + 1) * bs[0],
                        bp[1] * bs[1] : (bp[1] + 1) * bs[1],
                    ]
                    block = dbsm[
                        b[0] * bs[0] : (b[0] + 1) * bs[0],
                        b[1] * bs[1] : (b[1] + 1) * bs[1],
                    ]
                    self.assertTrue((block0 == block).all())

            # Check that empty positions are indeed empty
            block_mask = bsm.block_mask_build(None)

            if verbose:
                print(block_mask)

            block_mask = block_mask.repeat_interleave(32, dim=0).repeat_interleave(32, dim=1).float()
            self.assertEqual((dbsm * (1 - block_mask)).abs().sum(), 0)

            # Part 2: check multiplication behaviour
            a = torch.randn((1, size[1]), device=bsm.data.device).abs()

            c = bsm.reverse_matmul(a, transpose=True)
            c_0 = a.matmul(dbsm.t())

            # Basic check
            all_compare = torch.isclose(c, c_0)
            if not all_compare.all():
                # print((all_compare != True).nonzero(as_tuple=False))
                # print((c-c_0).abs().max())
                self.assertTrue(False)

            # Check matmul with sparse support
            b = torch.randn((1, size[0]), device=bsm.data.device).abs()

            bsm.matmul_with_output_sparse_support(b, a, overwrite_data=True)
            dbsm_back = bsm.to_dense()
            dbsm0_back = b.t().mm(a)
            dbsm0_back = dbsm0_back * bsm.to_dense(data_replace=torch.ones_like(bsm.data))

            self.assertTrue(dbsm0_back.isclose(dbsm_back).all())


if __name__ == "__main__":
    unittest.main()
