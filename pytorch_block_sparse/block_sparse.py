import numpy
import torch
import torch.nn


class BlockSparseMatrixBase(torch.nn.Module):
    # cols is a list of nonzero block column indexes (int32)
    # row_start is a index into cols (int32)
    # Data is (len(cols), block_shape, block_shape)
    def __init__(self, shape, block_mask, data, block_shape=(32, 32)):
        super(BlockSparseMatrixBase, self).__init__()
        self.int_type = torch.int32

        if len(shape) != 2:
            raise Exception("shape should be a tuple of 2 ints")

        self.shape = torch.Size(shape)
        if len(block_shape) != 2:
            raise Exception("block_shape should be a tuple of 2 ints")

        self.block_shape = tuple(block_shape)

        self.data = torch.nn.Parameter(data)

        self.rebuild(block_mask, callback=False)

    def updated_data(self):
        pass

    def get_differentiable_data(self):
        return self.data

    def rebuild(self, block_mask, block_ptr=None, callback=True):
        data = self.data
        block_shape = self.block_shape

        if block_mask.device != data.device:
            raise Exception(
                "block_mask and data should have same device, got %s and %s" % (block_mask.device, data.device)
            )

        blocks, cols_a, row_start_ends_a, rows_b, col_start_ends_b = self.build_indices(block_mask, block_ptr)

        if len(data.shape) != 2:
            raise Exception("data should be bidimensional, not of shape %s" % data.shape)
        if data.shape[0] != cols_a.shape[0] * block_shape[0]:
            raise Exception(
                "data.shape[0] (%d) should be equal to cols.shape[0]*block_shape[0] (%d)"
                % (data.shape[0], cols_a.shape[0] * block_shape[0])
            )
        if data.shape[1] != block_shape[1]:
            raise Exception(
                "data.shape[1] (%d) should be equal to block_shape[1] (%d)" % (data.shape[1], block_shape[1])
            )
        if data.dtype != torch.float32:
            raise Exception("data should be float32, not of type %s" % data.dtype)

        for name in (
            "cols_a",
            "row_start_ends_a",
            "rows_b",
            "col_start_ends_b",
            "blocks",
        ):
            self.register_buffer(name, locals()[name])

        self.sanity_check(self.cols_a, self.row_start_ends_a, self.shape, self.block_shape)
        self.sanity_check(
            self.rows_b,
            self.col_start_ends_b,
            (self.shape[1], self.shape[0]),
            (self.block_shape[1], self.block_shape[0]),
        )
        self.check_ = False
        if callback:
            self.updated_data()

    @staticmethod
    def blocks_count_(shape, block_shape):
        return torch.Size((shape[0] // block_shape[0], shape[1] // block_shape[1]))

    def blocks_count(self):
        return self.blocks_count_(self.shape, self.block_shape)

    def nnz_block_count(self):
        return self.blocks.shape[0] // 2

    def build_indices_(self, block_mask, block_ptr, nnzt, transpose_indices):
        device = block_mask.device
        X, Y = self.blocks_count()

        rows = nnzt[0]
        cols = nnzt[1]

        if transpose_indices:
            block_indices = torch.zeros(X * Y, dtype=torch.long, device=device)
            positions = rows * Y + cols
            # Set the index of used blocks at the used blocks positions : rest will stay zero
            # Add 1 temporarily to use 0 as a special value
            block_indices[positions] = block_ptr + 1
            # Reorganize the indexes with transposed ordering
            block_indices = block_indices.reshape(X, Y).t().reshape(X * Y)
            # Only keeps the non zero, and substract 1 to find back the right block index
            block_ptr = block_indices[torch.nonzero(block_indices, as_tuple=False)] - 1
            # Remove spurious dimension
            block_ptr = block_ptr.squeeze(-1)

            X, Y = Y, X

            rows = cols

            nnztt = torch.nonzero(block_mask.t(), as_tuple=False)
            cols = nnztt[:, 1]

        row_start_ends = torch.zeros((X + 1,), dtype=torch.long, device=device)

        row_start_ends.index_add_(
            0,
            rows + 1,
            torch.ones(size=(cols.shape[0],), dtype=torch.long, device=device),
        )
        row_start_ends = row_start_ends.cumsum(0).to(dtype=self.int_type)

        cols = torch.stack([cols, block_ptr], 1).to(dtype=self.int_type)

        return cols, row_start_ends

    def block_positions(self):
        return self.blocks.view(-1, 2).flip(-1)

    def build_indices(self, block_mask, block_ptr=None):
        # block_mask is a boolean mask that gives the block places
        # block_ptr, if not None, gives the block position in data for each element of cols_a, otherwise
        # assume that the content of block_ptr is just from 0..n_blocks
        # Used to recycle blocks

        nnz = torch.nonzero(block_mask, as_tuple=False)

        if block_ptr is None:
            block_ptr = torch.arange(0, nnz.shape[0], device=block_mask.device)

        # Sort the nnz according to block_ptr to build the self.blocks
        # data used by matmul_with_output_sparse_support_
        _, indices = block_ptr.sort()
        blocks = nnz[indices]

        blocks = blocks.flip(-1).flatten().to(dtype=self.int_type)

        nnzt = nnz.transpose(0, 1)
        cols_a, row_start_ends_a = self.build_indices_(block_mask, block_ptr, nnzt, False)
        rows_b, col_start_ends_b = self.build_indices_(block_mask, block_ptr, nnzt, True)

        verbose = False

        if verbose:
            print(f"row_start_ends_a=\n {self.row_start_ends_a}\ncols_a=\n {self.cols_a}\n")
            print(f"col_start_ends_b=\n {self.col_start_ends_b}\nrows_b=\n {self.rows_b}\n")

        return blocks, cols_a, row_start_ends_a, rows_b, col_start_ends_b

    def block_norm(self, p=2):
        d = self.data.reshape(-1, self.block_shape[0], self.block_shape[1])
        return torch.norm(d, p=p, dim=(1, 2), keepdim=False, out=None, dtype=None)

    def block_mask_build(self, block_positions):
        if block_positions is None:
            block_positions = self.build_coo_block_index().long()
        block_mask = torch.zeros(self.blocks_count(), dtype=torch.bool, device=self.data.device)

        block_mask[block_positions[0], block_positions[1]] = True
        return block_mask

    def block_replace(self, block_replacements):
        """block_replacements contains tuples : row, col, block_index in self.data"""
        # block positions contains 2d block coordinates
        block_positions = self.build_coo_block_index().long()
        block_replacements = block_replacements.to(dtype=torch.long).to(device=block_positions.device)
        # Get the column count
        XSIZE, YSIZE = self.blocks_count()

        if self.check_:
            # Make sure the block in block_replacements were not already assigned a block (block mask should be zero
            # at this position)
            # to conserve the one to one mapping between non_zero positions and self.data
            # Make sure too that new blocks are not mentionned twice in block_replacements
            # Build the block_mask
            block_mask0 = self.block_mask_build(block_positions)
            # Flatten block_mask
            block_mask = block_mask0.reshape(-1).long()

            # Build flat index of block_replacements
            new_positions = block_replacements[:, 0] * YSIZE + block_replacements[:, 1]
            ones = torch.ones_like(new_positions)
            # Increase the block count at new_positions -> should be one at most, meaning this position was not
            # assigned a block before this.
            block_mask.index_add_(0, new_positions, ones)

            # Check if we do not have a single reference count > 1
            block_mask = block_mask.reshape(block_mask0.shape)

            argmax = block_mask.argmax()
            argmax_x, argmax_y = argmax // YSIZE, argmax % YSIZE

            if block_mask[argmax_x, argmax_y] > 1:
                raise Exception(f"Block position ({argmax_x},{argmax_y}) was already used")

        # Do the block replacement
        replace_index = block_replacements[:, 2]
        replace_data = block_replacements[:, :2].t()
        block_positions[:, replace_index] = replace_data

        # Build the new block_mask
        block_mask = self.block_mask_build(block_positions)

        # Create a flat view of the block positions (contiguous in the PyTorch sense)
        block_positions_indices = block_positions[0] * YSIZE + block_positions[1]

        # Sort the positions
        _, indices = block_positions_indices.sort()

        # Get the 1D blocks indices in self.data
        block_ptr = indices

        # Rebuild all the structures using the block_mask and block_ptr info
        self.rebuild(block_mask, block_ptr)

    @classmethod
    def zeros(cls, shape, n_blocks=None, blocks=None, block_shape=(32, 32), device="cuda"):
        for i in range(2):
            if shape[i] % block_shape[i] != 0:
                raise Exception(f"Invalid shape: shape[{i}]={shape[i]} %% block_shape[{i}]={block_shape[i]} is not 0.")

        X, Y = cls.blocks_count_(shape, block_shape)

        if n_blocks is None:
            if blocks is not None:
                for b in blocks:
                    for i in range(2):
                        if b[i] * block_shape[i] >= shape[i]:
                            raise Exception(
                                f"Invalid block definition: block[{i}] = {b[i]} : should be < {shape[i] // block_shape[i]}"
                            )
                n_blocks = len(blocks)
            else:
                n_blocks = X * Y
        else:
            assert blocks is None or len(blocks) == n_blocks
        if len(shape) != 2 or shape[0] % block_shape[0] != 0 or shape[1] % block_shape[1] != 0:
            raise Exception("shape should be a tuple of 2 multiples of block_shape")

        if n_blocks > X * Y:
            raise Exception("Too many blocks : %d > %d * %d = %d" % (n_blocks, X, Y, X * Y))

        if not isinstance(blocks, torch.Tensor):
            if blocks is not None:
                positions = numpy.array(list(map(lambda b: b[0] * Y + b[1], blocks)))
            else:
                positions = numpy.random.choice(X * Y, size=n_blocks, replace=False)
            positions = torch.tensor(positions, dtype=torch.int64, device=device).sort()[0]
        else:
            positions = blocks.view(-1, 2).t()
            positions = positions[0] * Y + positions[1]

        block_mask = torch.zeros(X * Y, dtype=torch.bool, device=device)
        block_mask[positions] = True
        block_mask = block_mask.view(X, Y)
        data = torch.zeros(
            (n_blocks * block_shape[0], block_shape[1]),
            dtype=torch.float,
            device=device,
        )

        return cls(shape, block_mask, data, block_shape)

    @classmethod
    def ones(
        cls,
        shape,
        n_blocks=None,
        blocks=None,
        block_shape=(32, 32),
        device="cuda",
        positive=False,
    ):
        ret = cls.zeros(shape, n_blocks, blocks, block_shape, device)
        with torch.no_grad():
            ret.data += 1
        ret.updated_data()
        return ret

    @classmethod
    def randn(
        cls,
        shape,
        n_blocks=None,
        blocks=None,
        block_shape=(32, 32),
        device="cuda",
        positive=False,
    ):
        ret = cls.zeros(shape, n_blocks, blocks, block_shape, device)
        with torch.no_grad():
            if positive:
                ret.data.normal_().abs_()
            else:
                ret.data.normal_()
        ret.updated_data()
        return ret

    @classmethod
    def from_dense(cls, dense, block_shape=(32, 32), block_count=None, blocks=None, out=None, slow=False, exact=True):
        dense_block_count = (dense.shape[0] * dense.shape[1]) // (block_shape[0] * block_shape[1])
        dense0 = dense

        dshape = dense.shape
        bs = block_shape
        dense = dense.reshape(dshape[0] // bs[0], bs[0], dshape[1] // bs[1], bs[1])
        dense = dense.transpose(1, 2)

        if exact:
            assert block_count is None
            assert blocks is None
            assert out is None

            dense_mask = dense != 0
            block_mask = dense_mask.sum(2).sum(2) != 0
            block_count = block_mask.sum().item()
            blocks = block_mask.nonzero(as_tuple=False)

        if out is None:
            if blocks is None:
                if block_count is None:
                    block_count = dense_block_count
            else:
                block_count = len(blocks)

            ret = cls.zeros(
                dshape,
                n_blocks=block_count,
                block_shape=block_shape,
                blocks=blocks,
                device=dense.device,
            )
        else:
            ret = out

        if blocks is not None or block_count == dense_block_count:
            if slow:
                coo = ret.build_coo_block_index().long()
                # Legacy version, used for testing only
                for i in range(coo.shape[1]):
                    r, c = coo[0][i], coo[1][i]
                    bs = ret.block_shape
                    part = dense0[r * bs[0] : (r + 1) * bs[0], c * bs[1] : (c + 1) * bs[1]]
                    part = part.t().reshape(block_shape[0], block_shape[1])

                    with torch.no_grad():
                        ret.data[i * bs[0] : (i + 1) * bs[0]] = part
            else:
                if blocks is None:
                    blocks = ret.block_positions().long()
                if not isinstance(blocks, torch.Tensor):
                    blocks = torch.tensor(blocks, dtype=torch.long, device=ret.data.device)

                block_mask_nnz = tuple(blocks.t())
                extract = dense[block_mask_nnz].transpose(1, 2)
                extract = extract.reshape(extract.shape[0] * extract.shape[2], extract.shape[1])

                with torch.no_grad():
                    ret.data.copy_(extract)
        else:
            # We just keep the first elements in the dense matrix
            # Of course this only captures the statistical distribution in the dense matrix
            param_count = ret.data.numel()
            with torch.no_grad():
                ret.data.copy_(dense.flatten()[:param_count].reshape(ret.data.shape))

        ret.updated_data()

        return ret

    def __repr__(self):
        return "%s(shape=%s, cols=%s, row_start_ends_a=%s, data=%s, block_shape=%s)" % (
            self.__class__.__name__,
            self.shape,
            self.cols_a.shape,
            self.row_start_ends_a.shape,
            self.data.shape,
            self.block_shape,
        )

    def multiply_(self, factor):
        with torch.no_grad():
            self.data.multiply_(factor)
        self.updated_data()

    def build_coo_block_index(self):
        device = self.cols_a.device
        # Build a tensor to store the row indices.
        # It's one element too long for the moment, we'll trim it later
        rows = torch.zeros((self.cols_a.shape[0] + 1), dtype=self.int_type, device=device)

        # Change self.row_start_ends_a to the right type
        row_end_prepare = self.row_start_ends_a[1:].long()

        # Add ones to the start position of each new row
        rows.index_add_(
            0,
            row_end_prepare,
            torch.ones(size=row_end_prepare.shape, dtype=self.int_type, device=device),
        )

        # Accumulate those start positions to fill the remaining positions
        rows = rows.cumsum(0).to(dtype=self.int_type)

        # Trim the last element: it's just a left over
        rows = rows[:-1].unsqueeze(-1)

        # Build the coo indexes
        summary = torch.cat([rows, self.cols_a], -1)

        _, indices = summary[:, 2].sort()
        summary = summary[indices]
        if self.check_:
            compare = summary[:, 2] == torch.arange(0, summary.shape[0], device=summary.device)
            assert compare.all()

        summary = summary.t()
        return summary[:2]

    def to_sparse(self, data_replace=None):
        coo = self.build_coo_block_index().long()

        if data_replace is not None:
            data = data_replace
        else:
            data = self.data
        data = data.reshape(-1, self.block_shape[1], self.block_shape[0])
        data = data.transpose(1, 2)
        out = torch.sparse.FloatTensor(
            coo,
            data,
            (self.shape[0] // self.block_shape[0], self.shape[1] // self.block_shape[1])
            + (self.block_shape[0], self.block_shape[1]),
        )

        return out

    def to_dense(self, data_replace=None):
        out = self.to_sparse(data_replace)
        out = out.to_dense()
        out = out.transpose(1, 2)
        out = out.reshape(self.shape[0], self.shape[1])

        return out

    def sanity_check(self, cols, row_end, shape, block_shape):
        row_end = row_end[1:]
        if len(cols.shape) != 2:
            raise Exception("cols should be bidimensional, not of shape %s" % cols.shape)
        if cols.dtype != self.int_type:
            raise Exception("cols should be %s, not of type %s" % (self.int_type, cols.dtype))
        max_col = cols[:, 0].max()
        if max_col > shape[1] / block_shape[1]:
            raise Exception(
                "cols max element (%d) cannot be larger than shape[1]/block_shape[1] (%d)"
                % (max_col, shape[1] / block_shape[1])
            )

        if len(row_end.shape) != 1:
            raise Exception("row_end should be unidimensional, not of shape %s" % row_end.shape)
        if row_end.shape[0] != shape[0] / block_shape[0]:
            raise Exception(
                "row_end.shape[0] (%d) should be equal to shape[0]/block_shape[0] (%d)"
                % (row_end.shape[0], shape[0] / block_shape[0])
            )
        if row_end.dtype != self.int_type:
            raise Exception("row_end should be %s, not of type %s" % (self.int_type, row_end.dtype))

        max_row_end = row_end.max()
        if max_row_end > cols.shape[0]:
            raise Exception(
                "row_end max element (%d) cannot be larger than cols count (%d)" % (max_row_end, self.cols.shape[0])
            )
        last_row_end = row_end[-1]
        if last_row_end != cols.shape[0]:
            raise Exception(
                "row_end last element (%d) should be equal to cols count (%d)" % (last_row_end, self.cols.shape[0])
            )

    def check_with_dense(self, dense_version):
        # Partial check of to_dense
        coo = self.build_coo_block_index().long()

        for i in range(coo.shape[1]):
            r, c = coo[0][i], coo[1][i]
            bs = self.block_shape
            from_sparse = self.data[i * bs[0] : (i + 1) * bs[0], :].t()
            from_dense = dense_version[r * bs[0] : (r + 1) * bs[0], c * bs[1] : (c + 1) * bs[1]]
            if not (from_sparse == from_dense).all():
                print(f"r={r},c={c}\n", from_sparse[::8, ::8], "\n", from_dense[::8, ::8])
                raise Exception("non matching data")

    # Prepare the data itself. This function does not deal at all with true matrix dimensions, you have to check them
    # independently.
    def tensor_prepare(self, t, message, transpose):
        """Return prepared tensor, should we transpose it in CUDA kernel"""
        ret = None
        if t.is_contiguous():
            ret = [t, True]
        if t.t().is_contiguous():
            ret = [t, False]

        if ret is not None:
            if transpose:
                ret[1] = not ret[1]
            return ret

        # warnings.warn(message)
        return t.contiguous(), False

    def reverse_matmul_(self, dense_a, transpose=True):
        """Compute a.matmul(self.t()) if transposed, else a.matmul(self)"""
        import block_sparse_native

        if dense_a.dim() > 2:
            dense_a = dense_a.reshape(-1, dense_a.shape[-1])

        shape_a = list(dense_a.shape)
        shape_b = [self.shape[0], self.shape[1]]
        block_shape = list(self.block_shape)

        if transpose:
            shape_b.reverse()
            block_shape.reverse()

        if shape_a[1] != shape_b[0]:
            raise Exception(
                "Invalid matrices sizes (%d, %d) x (%d, %d)" % (shape_a[0], shape_a[1], shape_b[0], shape_b[1])
            )

        out = torch.zeros((shape_b[1], shape_a[0]), device=dense_a.device)

        if transpose:
            ptr_b = self.row_start_ends_a
            indices_b = self.cols_a
            dim = 0
        else:
            ptr_b = self.col_start_ends_b
            indices_b = self.rows_b
            dim = 1

        assert (shape_a[1] % block_shape[1]) == 0
        assert self.data.is_contiguous()
        assert out.is_contiguous()

        assert ptr_b.is_contiguous()
        assert ptr_b.dtype == self.int_type
        assert indices_b.is_contiguous()
        assert indices_b.dtype == self.int_type

        assert ptr_b.shape[0] == self.blocks_count()[dim] + 1

        if transpose:
            data_b = self.data
        else:
            # TEMPORARY : move this to kernel
            data = self.data.view(-1, *block_shape)
            data = data.transpose(1, 2)
            data_b = data.reshape(-1, block_shape[1]).contiguous()

        if not dense_a.is_contiguous():
            # warnings.warn(f"pytorch_block_sparse.BlockSparseMatrix.reverse_matmul:"
            #               f" DEGRADED performance, dense_a is not contiguous {dense_a.stride()}")
            dense_a = dense_a.contiguous()

        verbose = False
        if verbose:
            print(
                "reverse_matmul\ndense_a=\n",
                dense_a[::32, ::32],
                "\nptr_b=\n",
                ptr_b,
                "\nindices_b=\n",
                indices_b,
                "\ndata_b=\n",
                data_b[::32, ::32],
            )
            print("reverse_matmul_\n", dense_a.shape, data_b.shape)

        block_sparse_native.blocksparse_matmul_cutlass(
            dense_a,
            True,
            ptr_b,
            indices_b,
            data_b,
            dense_a.shape[0],
            shape_b[1],
            shape_b[0],
            block_shape[1],
            block_shape[0],
            out,
        )
        return out.t()

    def flatten_first_dims(self, dense_a):
        if dense_a.dim() < 2:
            raise Exception(
                f"Invalid dimensions for dense_a {dense_a.shape} : dense_a should have at least 2 dimensions."
            )
        rewritten_a = dense_a
        if dense_a.dim() > 2:
            rewritten_a = dense_a.reshape(-1, dense_a.shape[-1])

        return rewritten_a, dense_a.shape[:-1]

    def unflatten_first_dims(self, result, info):
        shape_start = info
        if len(shape_start) > 1:
            result = result.view(*shape_start, result.shape[-1])
        return result

    def reverse_matmul(self, dense_a, transpose):
        assert str(self.data.device).startswith("cuda")

        rewritten_a, info_a = self.flatten_first_dims(dense_a)
        ret = self.reverse_matmul_(rewritten_a, transpose=transpose)

        ret = self.unflatten_first_dims(ret, info_a)
        return ret

    def matmul_with_output_sparse_support_(self, dense_a, dense_b, overwrite_data=False):
        """Compute  c = a.t().mm(b) where c is sparse (we just keep the results where c is non_zero)."""
        import block_sparse_native

        shape_a = dense_a.shape
        shape_b = dense_b.shape
        shape_c = self.shape

        # Check that sizes are compatible for a.t().mm(b)
        assert shape_a[0] == shape_b[0]
        assert shape_c[0] == shape_a[1]
        assert shape_c[1] == shape_b[1]

        blocks = self.blocks
        blocks_len = blocks.shape[0] // 2
        block_shape = self.block_shape

        assert (shape_a[1] % block_shape[1]) == 0
        assert (shape_b[1] % block_shape[0]) == 0

        if overwrite_data:
            data = self.data
        else:
            data = torch.zeros_like(self.data)

        message = (
            "pytorch_block_sparse.BlockSparseMatrix.matmul_with_output_sparse_support:"
            " DEGRADED performance, dense_%s is not contiguous"
        )
        prepared_a, transpose_a = self.tensor_prepare(dense_a, message % "a", True)
        prepared_b, transpose_b = self.tensor_prepare(dense_b, message % "b", False)

        # We interpret a as transposed, so we pass shape_a[1], shape_a[0] as a shape, and transpose_a
        # will be set correcly too (for a "normal" contiguous pytorch matrix a, transpose_a will be true)
        block_sparse_native.blocksparse_matmul_back_cutlass(
            prepared_a,
            transpose_a,
            prepared_b,
            transpose_b,
            shape_a[1],
            shape_b[1],
            shape_a[0],
            self.block_shape[0],
            self.block_shape[1],
            data,
            blocks,
            blocks_len,
        )

        return data

    def matmul_with_output_sparse_support(self, dense_a, dense_b, overwrite_data=False):
        assert str(self.data.device).startswith("cuda")

        rewritten_a, info_a = self.flatten_first_dims(dense_a)
        rewritten_b, info_b = self.flatten_first_dims(dense_b)
        assert info_a == info_b
        ret = self.matmul_with_output_sparse_support_(rewritten_a, rewritten_b, overwrite_data)

        return ret


class BlockSparseMatrix(BlockSparseMatrixBase):
    # cols is a list of nonzero block column indexes (int32)
    # row_start is a index into cols (int32)
    # Data is (len(cols), block_shape, block_shape)
    def __init__(self, shape, block_mask, data, block_shape=(32, 32)):
        if len(shape) != 2 or shape[0] % 32 != 0 or shape[1] % 32 != 0:
            raise Exception("shape should be a tuple of 2 multiples of 32")

        if len(block_shape) != 2 or block_shape[0] % 32 != 0 or block_shape[1] % 32 != 0:
            raise Exception("block_shape should be a tuple of 2 multiples of 32")

        super(BlockSparseMatrix, self).__init__(shape, block_mask, data, block_shape)


class BlockSparseMatrixEmulator(BlockSparseMatrixBase):
    # cols is a list of nonzero block column indexes (int32)
    # row_start is a index into cols (int32)
    # Data is (len(cols), block_shape, block_shape)
    def __init__(self, shape, block_mask, data, block_shape):
        super(BlockSparseMatrixEmulator, self).__init__(shape, block_mask, data, block_shape)
        self.register_parameter("_dense", None)
        self.updated_data()

    def get_differentiable_data(self):
        return self._dense

    def to_dense(self, data_replace=None):
        if data_replace is None:
            return self._dense
        return data_replace * self._mask

    def _update_data_from_dense(self):
        _ = self.from_dense(self._dense, out=self, exact=False)

    def updated_data(self):
        with torch.no_grad():
            self._dense = torch.nn.Parameter(super().to_dense())
            self._mask = super().to_dense(data_replace=torch.ones_like(self.data)) == 1

    def reverse_matmul(self, dense_a, transpose=True):
        m = self._dense.t() if transpose else self._dense
        mask = self._mask.t() if transpose else self._mask
        return dense_a.matmul(m * mask)  # The self._mask multiplication is not really needed, but ...

    def matmul_with_output_sparse_support(self, dense_a, dense_b, overwrite_data=False):
        """Compute  c = a.t().mm(b) where c is sparse (we just keep the results where c is non_zero)."""
        return dense_a.t().mm(dense_b) * self._mask
