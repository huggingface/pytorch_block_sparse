import torch
import torch.nn
import numpy
import warnings

class BlockSparseMatrix(torch.nn.Module):
    # cols is a list of nonzero block column indexes (int32)
    # row_start is a index into cols (int32)
    # Data is (len(cols), block_shape, block_shape)
    def __init__(self, shape, block_mask, data, block_shape=(16, 16)):
        super(BlockSparseMatrix, self).__init__()

        if block_mask.device != data.device:
            raise Exception("block_mask and data should have same device, got %s and %s" % (block_mask.device, data.device))

        if len(shape) != 2 or shape[0] % 16 != 0 or shape[1] % 16 != 0:
            raise Exception("shape should be a tuple of 2 multiples of 16")
        self.shape = torch.Size(shape)
        if len(block_shape) != 2 or block_shape[0] % 16 != 0 or block_shape[1] % 16 != 0:
            raise Exception("block_shape should be a tuple of 2 multiples of 16")
        self.block_shape = tuple(block_shape)

        blocks, cols_a, row_start_ends_a, rows_b, col_start_ends_b = self.build_indices(block_mask)

        if len(data.shape) != 2:
            raise Exception("data should be bidimensional, not of shape %s" % data.shape)
        if data.shape[0] != cols_a.shape[0] * block_shape[0]:
            raise Exception("data.shape[0] (%d) should be equal to cols.shape[0]*block_shape[0] (%d)" % (data.shape[0], cols_a.shape[0] * block_shape[0]))
        if data.shape[1] !=  block_shape[1]:
            raise Exception("data.shape[1] (%d) should be equal to block_shape[1] (%d)" % (data.shape[1], block_shape[1]))
        if data.dtype != torch.float32:
            raise Exception("data should be float32, not of type %s" % data.dtype)


        self.data = torch.nn.Parameter(data)
        for name in ("block_mask", "cols_a", "row_start_ends_a", "rows_b", "col_start_ends_b", "blocks"):
            self.register_buffer(name, locals()[name])

        self.sanity_check(self.cols_a, self.row_start_ends_a, self.shape, self.block_shape)
        self.sanity_check(self.rows_b, self.col_start_ends_b, (self.shape[1], self.shape[0]), (self.block_shape[1], self.block_shape[0]))

    @staticmethod
    def blocks_count_(shape, block_shape):
        return torch.Size((shape[0] // block_shape[0], shape[1] // block_shape[1]))

    def blocks_count(self):
        return self.blocks_count_(self.shape, self.block_shape)

    def build_indices_(self, block_mask, nnzt, transpose_indices):
        device = block_mask.device
        X, Y = self.blocks_count_(self.shape, self.block_shape)

        rows = nnzt[0]
        cols = nnzt[1]

        block_shuffle = torch.arange(0, cols.shape[0], device = device)

        if transpose_indices:
            block_indices = torch.zeros(X*Y, dtype=torch.long, device = device)
            positions = rows * Y + cols
            # Set the index of used blocks at the used blocks positions : rest will stay zero
            # Add 1 temporarily to use 0 as a special value
            block_indices[positions] = block_shuffle + 1
            # Reorganize the indexes with transposed ordering
            block_indices = block_indices.reshape(X, Y).t().reshape(X * Y)
            # Only keeps the non zero, and substract 1 to find back the right block index
            block_shuffle = block_indices[block_indices.nonzero()] - 1
            # Remove spurious dimension
            block_shuffle = block_shuffle.squeeze(-1)

            X, Y = Y, X

            rows = cols

            nnztt = block_mask.t().nonzero()
            cols = nnztt[:,1]

        row_start_ends = torch.zeros((X + 1,), dtype=torch.long, device = device)

        row_start_ends.index_add_(0, rows + 1, torch.ones(size=(cols.shape[0],), dtype=torch.long, device = device))
        row_start_ends = row_start_ends.cumsum(0).int()

        cols = torch.stack([cols, block_shuffle], 1).int()

        return cols, row_start_ends

    def build_indices(self, block_mask):
        nnz = block_mask.nonzero()
        blocks = nnz.flip(-1).flatten().to(dtype=torch.int32)

        nnzt = nnz.transpose(0, 1)
        cols_a, row_start_ends_a = self.build_indices_(block_mask, nnzt, False)
        rows_b, col_start_ends_b = self.build_indices_(block_mask, nnzt, True)

        verbose = False

        if verbose:
            print(f"row_start_ends_a=\n {self.row_start_ends_a}\ncols_a=\n {self.cols_a}\n")
            print(f"col_start_ends_b=\n {self.col_start_ends_b}\nrows_b=\n {self.rows_b}\n")

        return blocks, cols_a, row_start_ends_a, rows_b, col_start_ends_b

    @classmethod
    def zeros(cls, shape, n_blocks = None, blocks = None, block_shape=(32, 32), device = "cuda"):
        assert(str(device).startswith("cuda"))
        for i in range(2):
            if shape[i] % block_shape[i] != 0:
                raise Exception(f"Invalid shape: shape[{i}]({shape[i]}) %% block_shape[{i}]({block_shape[i]}) is not 0.")
        if n_blocks == None:
            assert(blocks != None)
            for b in blocks:
                for i in range(2):
                    if b[i] * block_shape[i] >= shape[i]:
                        raise Exception(f"Invalid block definition: block[{i}] = {b[i]} : should be < {shape[i] // block_shape[i]}")
            n_blocks = len(blocks)
        else:
            assert(blocks == None)
        if len(shape) != 2 or shape[0] % block_shape[0] != 0 or shape[1] % block_shape[1] != 0:
            raise Exception("shape should be a tuple of 2 multiples of block_shape")

        X, Y = cls.blocks_count_(shape, block_shape)

        if n_blocks > X * Y:
            raise Exception("Too many blocks : %d > %d * %d = %d" % (n_blocks, X, Y, X * Y))
        if blocks != None:
            positions = numpy.array(list(map(lambda b : b[0] * Y + b[1], blocks)))
        else:
            positions = numpy.random.choice(X*Y, size=n_blocks, replace=False)
        positions = torch.tensor(positions, dtype=torch.int64, device = device).sort()[0]

        block_mask = torch.zeros(X * Y, dtype=torch.bool, device = device)
        block_mask[positions] = True
        block_mask = block_mask.view(X, Y)
        data = torch.zeros((n_blocks * block_shape[0], block_shape[1]), dtype=torch.float, device = device)

        return cls(shape, block_mask, data, block_shape)

    @classmethod
    def randn(cls, shape, n_blocks, blocks = None, block_shape=(32, 32), device = "cuda"):
        ret = cls.zeros(shape, n_blocks, blocks, block_shape, device)
        with torch.no_grad():
            ret.data.normal_()
        return ret

    @classmethod
    def from_dense(cls, dense, block_shape = (32, 32), block_count = None):
        dense_block_count = (dense.shape[0] * dense.shape[1]) // (block_shape[0] * block_shape[1])
        if block_count == None:
            block_count = dense_block_count

        ret = cls.zeros(dense.shape, n_blocks = block_count, block_shape = block_shape, device = dense.device)

        if block_count == dense_block_count:
            # TODO : use some pytorch dimensions transposition to speed up this block by block copy
            coo = ret.build_coo_block_index().long()

            for i in range(coo.shape[1]):
                r, c = coo[0][i], coo[1][i]
                bs = ret.block_shape
                ret.data[i * bs[0]:(i + 1) * bs[0], :] = dense[r * bs[0]:(r + 1) * bs[0], c * bs[1]:(c + 1) * bs[1]].t()
        else:
            param_count = ret.data.numel()
            density = block_count / dense_block_count
            ret.data.copy_(dense.flatten()[:param_count].reshape(ret.data.shape) / density)

        return ret

    def __repr__(self):
        return "%s(shape=%s, cols=%s, row_start_ends_a=%s, data=%s, block_shape=%s)" % (self.__class__.__name__,
                                                                               self.shape,
                                                                               self.cols_a.shape,
                                                                               self.row_start_ends_a.shape,
                                                                               self.data.shape,
                                                                               self.block_shape)

    def build_coo_block_index(self):
        device = self.cols_a.device
        # Build a tensor to store the row indices.
        # It's one element too long for the moment, we'll trim it later
        rows = torch.zeros((self.cols_a.shape[0] + 1), dtype=torch.int32, device=device)

        # Change self.row_start_ends_a to the right type
        row_end_prepare = self.row_start_ends_a[1:].long()

        # Add ones to the start position of each new row
        rows.index_add_(0, row_end_prepare, torch.ones(size=row_end_prepare.shape, dtype=torch.int32, device=device))

        # Accumulate those start positions to fill the remaining positions
        rows = rows.cumsum(0).int()

        # Trim the last element: it's just a left over
        rows = rows[:-1]

        # Build the coo indexes
        return torch.stack([rows, self.cols_a[:,0]], 0)

    def to_sparse(self, data_replace = None):
        coo = self.build_coo_block_index().long()

        if data_replace is not None:
            data = data_replace
        else:
            data = self.data
        data = data.reshape(-1, *self.block_shape).transpose(1,2)
        out = torch.sparse.FloatTensor(coo, data,
                                       (self.shape[0] // self.block_shape[0], self.shape[1] // self.block_shape[1]) + self.block_shape)

        return out

    def to_dense(self, data_replace = None):
        out = self.to_sparse(data_replace)
        out = out.to_dense()
        out = out.transpose(1,2)
        out = out.reshape(self.shape[0], self.shape[1])

        return out

    def sanity_check(self, cols, row_end, shape, block_shape):
        row_end = row_end[1:]
        if len(cols.shape) != 2:
            raise Exception("cols should be bidimensional, not of shape %s" % cols.shape)
        if cols.dtype != torch.int32:

            raise Exception("cols should be int32, not of type %s" % cols.dtype)
        max_col = cols[:,0].max()
        if max_col > shape[1] / block_shape[1]:
            raise Exception("cols max element (%d) cannot be larger than shape[1]/block_shape[1] (%d)" % (max_col, shape[1] / block_shape[1]))

        if len(row_end.shape) != 1:
            raise Exception("row_end should be unidimensional, not of shape %s" % row_end.shape)
        if row_end.shape[0] != shape[0] / block_shape[0]:
            raise Exception("row_end.shape[0] (%d) should be equal to shape[0]/block_shape[0] (%d)" % (row_end.shape[0], shape[0] / block_shape[0]))
        if row_end.dtype != torch.int32:
            raise Exception("row_end should be int32, not of type %s" % row_end.dtype)

        max_row_end = row_end.max()
        if max_row_end > cols.shape[0]:
            raise Exception("row_end max element (%d) cannot be larger than cols count (%d)" % (max_row_end, self.cols.shape[0]))
        last_row_end = row_end[-1]
        if last_row_end != cols.shape[0]:
            raise Exception("row_end last element (%d) should be equal to cols count (%d)" % (last_row_end, self.cols.shape[0]))

    def check_with_dense(self, dense_version):
        # Partial check of to_dense
        coo = self.build_coo_block_index().long()

        for i in range(coo.shape[1]):
            r,c = coo[0][i], coo[1][i]
            bs = self.block_shape
            from_sparse = self.data[i * bs[0]:(i +1)* bs[0],:].t()
            from_dense = dense_version[r * bs[0]:(r+1)*bs[0], c * bs[1]:(c + 1)*bs[1]]
            if not (from_sparse == from_dense).all():
                print(f"r={r},c={c}\n", from_sparse[::8,::8], "\n", from_dense[::8,::8])
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

        if ret != None:
            if transpose:
                ret[1] = not ret[1]
            return ret

        #warnings.warn(message)
        return t.contiguous(), False

    def reverse_matmul_(self, dense_a, transpose = True):
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
            raise Exception("Invalid matrices sizes (%d, %d) x (%d, %d)" % (shape_a[0], shape_a[1], shape_b[0], shape_b[1]))

        out = torch.zeros((shape_b[1], shape_a[0]), device = dense_a.device)

        if transpose:
            ptr_b = self.row_start_ends_a
            indices_b = self.cols_a
            dim = 0
        else:
            ptr_b = self.col_start_ends_b
            indices_b = self.rows_b
            dim = 1

        assert((shape_a[1] % block_shape[1]) == 0)
        assert(self.data.is_contiguous())
        assert(out.is_contiguous())

        assert(ptr_b.is_contiguous())
        assert(ptr_b.dtype == torch.int32)
        assert(indices_b.is_contiguous())
        assert(indices_b.dtype == torch.int32)

        assert(ptr_b.shape[0] == self.blocks_count()[dim] + 1)

        if transpose:
            data_b = self.data
        else:
            # TEMPORARY : move this to kernel
            data = self.data.view(-1, *block_shape)
            data = data.transpose(1,2)
            data_b = data.reshape(-1, block_shape[1]).contiguous()

        if not dense_a.is_contiguous():
            #warnings.warn(f"pytorch_block_sparse.BlockSparseMatrix.reverse_matmul: DEGRADED performance, dense_a is not contiguous {dense_a.stride()}")
            dense_a = dense_a.contiguous()

        verbose = False
        if verbose:
            print("reverse_matmul\ndense_a=\n", dense_a[::32,::32],"\nptr_b=\n", ptr_b, "\nindices_b=\n", indices_b, "\ndata_b=\n", data_b[::32,::32])
            print("reverse_matmul_\n", dense_a.shape, data_b.shape)

        block_sparse_native.blocksparse_matmul_cutlass(dense_a,
                                                       True,
                                                       ptr_b, indices_b,
                                                       data_b,
                                                       dense_a.shape[0], shape_b[1], shape_b[0],
                                                       block_shape[1], block_shape[0],
                                                       out)
        return out.t()

    def flatten_first_dims(self, dense_a):
        if dense_a.dim() < 2:
            raise Exception(f"Invalid dimensions for dense_a {dense_a.shape} : dense_a should have at least 2 dimensions.")
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
        rewritten_a, info_a = self.flatten_first_dims(dense_a)
        ret = self.reverse_matmul_(rewritten_a, transpose = transpose)

        ret = self.unflatten_first_dims(ret, info_a)
        return ret

    def matmul_with_output_sparse_support_(self, dense_a, dense_b, overwrite_data = False):
        """Compute  c = a.t().mm(b) where c is sparse (we just keep the results where c is non_zero)."""
        import block_sparse_native
        shape_a = dense_a.shape
        shape_b = dense_b.shape
        shape_c = self.shape

        # Check that sizes are compatible for a.t().mm(b)
        assert(shape_a[0] == shape_b[0])
        assert(shape_c[0] == shape_a[1])
        assert(shape_c[1] == shape_b[1])

        blocks_len = self.blocks.shape[0] // 2
        block_shape = self.block_shape

        assert ((shape_a[1] % block_shape[1]) == 0)
        assert ((shape_b[1] % block_shape[0]) == 0)

        if overwrite_data:
            data = self.data
        else:
            data = torch.zeros_like(self.data)

        message = "pytorch_block_sparse.BlockSparseMatrix.matmul_with_output_sparse_support: DEGRADED performance, dense_%s is not contiguous"
        prepared_a, transpose_a = self.tensor_prepare(dense_a, message % "a", True)
        prepared_b, transpose_b = self.tensor_prepare(dense_b, message % "b", False)

        # We interpret a as transposed, so we pass shape_a[1], shape_a[0] as a shape,
        # and transpose_a will be set correcly too (for a "normal" contiguous pytorch matrix a, transpose_a will be true)
        block_sparse_native.blocksparse_matmul_back_cutlass(prepared_a, transpose_a, prepared_b, transpose_b,
                                                            shape_a[1], shape_b[1], shape_a[0],
                                                            self.block_shape[0], self.block_shape[1],
                                                            data,
                                                            self.blocks, blocks_len)

        return data

    def matmul_with_output_sparse_support(self, dense_a, dense_b, overwrite_data = False):
        rewritten_a, info_a = self.flatten_first_dims(dense_a)
        rewritten_b, info_b = self.flatten_first_dims(dense_b)
        assert(info_a == info_b)
        ret = self.matmul_with_output_sparse_support_(rewritten_a, rewritten_b, overwrite_data)

        return ret


