from unittest import TestCase
import torch
from torch import tensor
import unittest
from pytorch_block_sparse import BlockSparseMatrix

class TestFun(TestCase):

    def test0(self):
        tests = [dict(size= [128, 64],
                      blocks= [(0, 0), (1, 0), (2, 0), (0, 1), ],
                      row_start_ends_a=tensor([0, 2, 3, 4, 4]),
                      cols_a=tensor([[0, 0],[1, 1],[0, 2],[0, 3]]),
                      col_start_ends_b =tensor([0, 3, 4]),
                      rows_b= tensor([[0, 0], [1, 2], [2, 3],[0, 1]])
                      )
                 ]
        block_shape = (32, 32)
        device = "cuda"
        for test_info in tests:
            size = test_info["size"]
            blocks  =test_info["blocks"]
            bsm = BlockSparseMatrix.randn((size[0], size[1]), None, blocks=blocks, block_shape=block_shape, device=device)

            for key in test_info:
                if "row" in key or "col" in key:
                    bsm_a = getattr(bsm, key)
                    ref = test_info[key].to(device=device, dtype=torch.int32)
                    check = (bsm_a == ref).all()
                    if not check:
                        raise Exception(f"Non matching attribute {key}:\n{bsm_a}\n!=\n{ref} (ref).")

    def test1(self):
        sizes = [(32, 32), (64, 32), (32, 64), (64, 64), (256, 64)]
        for size in sizes:
            print(f"size={size}")
            block_shape = (32, 32)
            block_count  = size[0] * size[1] // (block_shape[0] * block_shape[1])
            device = "cuda"

            bsm = BlockSparseMatrix.randn(size, block_count,  block_shape=block_shape, device=device)
            a = bsm.to_dense()
            bsm.check_with_dense(a)

            bsm2 = BlockSparseMatrix.from_dense(a, block_shape, block_count = None)
            bsm2.check_with_dense(a)

            a2 = bsm2.to_dense()

            if not (a == a2).all():
                print((a == a2)[::8,::8])
                raise Exception("Non matching matrices, BlockSparseMatrix.from_dense is not correct.")

    def test2(self):
        bsm = BlockSparseMatrix.zeros((32, 32), 1, block_shape=(32,32), device="cuda")
        hash(bsm)

    def test_block_norm(self):
        nblocks = 6
        block_shape = (32, 32)
        bsm = BlockSparseMatrix.randn((256, 256), nblocks, block_shape=block_shape, device="cuda")
        n = bsm.block_norm()
        self.assertEqual(n.dim(), 1)
        self.assertEqual(n.shape[0], nblocks)

        d = bsm.data.reshape(-1, block_shape[0] * block_shape[1])
        d = (d*d).sum(-1).sqrt()
        self.assertTrue(d.isclose(n).all())

    def test_block_replace(self):
        tests = [dict(size=[128, 64],
                      blocks=[(0, 0), (1, 0), (2, 0), (0, 1), ],
                      block_info=[(0,0),(0,1),(1,0), (2,0)],
                      block_replace=[(3, 1, 0), (2,1,2), (1,1,1)], # row, col, block_index
                      after=dict(row_start_ends_a=[0, 0, 1, 3, 4],
                                 cols_a=[[1, 1],[0, 3],[1, 2],[1, 0]],
                                 block_mask=[[0, 0],[0,  1],[1,  1],[0,  1]]
                                 )
                      ),
                 dict(size=[128, 64],
                      blocks=[(0, 0), (1, 0), (2, 0), (0, 1), ],
                      block_info=[(0, 0), (0, 1), (1, 0), (2, 0)],
                      block_replace=[(0, 1, 0)],  # row, col, block_index
                      error = "Block position (0,1) was already used"
                      )
                 ]
        block_shape = (32, 32)
        device = "cuda"
        for test_info in tests:
            size = test_info["size"]
            blocks = test_info["blocks"]
            bsm = BlockSparseMatrix.randn((size[0], size[1]), None, blocks=blocks, block_shape=block_shape,
                                          device=device)
            bsm.check_ = True

            bsm.to_dense()
            block_positions = bsm.build_coo_block_index().t()
            for i, b in enumerate(test_info["block_info"]):
                block_position = tuple(block_positions[i].cpu().numpy())
                self.assertEqual(b, block_position)

            try:
                bsm.block_replace(test_info["block_replace"])
            except Exception as e:
                if test_info.get("error") == str(e):
                    continue
                raise

            for k,v in test_info["after"].items():
                if k != "block_mask":
                    r = getattr(bsm, k)
                else:
                    r = bsm.block_mask_build(None).long()
                v = torch.tensor(v, device = r.device)

                self.assertTrue((r==v).all())

        def test_to_sparse(self):
            pass



if __name__ == '__main__':
    unittest.main()
