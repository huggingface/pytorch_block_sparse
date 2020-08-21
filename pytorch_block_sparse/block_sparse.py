import torch
import numpy
import warnings


class BlockSparseMatrix:
    # cols is a list of nonzero block column indexes (int32)
    # row_start is a index into cols (int32)
    # Data is (len(cols), block_shape, block_shape)
    def __init__(self, shape, block_mask, data, block_shape=(16, 16)):
        if block_mask.device != data.device:
            raise Exception("block_mask and data should have same device, got %s and %s" % (block_mask.device, data.device))

        self.device = block_mask.device

        if len(shape) != 2 or shape[0] % 16 != 0 or shape[1] % 16 != 0:
            raise Exception("shape should be a tuple of 2 multiples of 16")
        self.shape = torch.Size(shape)
        if len(block_shape) != 2 or block_shape[0] % 16 != 0 or block_shape[1] % 16 != 0:
            raise Exception("block_shape should be a tuple of 2 multiples of 16")
        self.block_shape = block_shape

        self.block_mask = block_mask
        self.build_indices()

        if len(data.shape) != 2:
            raise Exception("data should be bidimensional, not of shape %s" % data.shape)
        if data.shape[0] != self.cols_a.shape[0] * block_shape[0]:
            raise Exception("data.shape[0] (%d) should be equal to cols.shape[0]*block_shape[0] (%d)" % (data.shape[0], self.cols_a.shape[0] * block_shape[0]))
        if data.shape[1] !=  block_shape[1]:
            raise Exception("data.shape[1] (%d) should be equal to block_shape[1] (%d)" % (data.shape[1], block_shape[1]))
        if data.dtype != torch.float32:
            raise Exception("data should be float32, not of type %s" % data.dtype)

        self.data = data

        self.sanity_check(self.cols_a, self.row_start_ends_a, self.shape, self.block_shape)
        self.sanity_check(self.rows_b, self.col_start_ends_b, (self.shape[1], self.shape[0]), (self.block_shape[1], self.block_shape[0]))

    @staticmethod
    def blocks_count_(shape, block_shape):
        return torch.Size((shape[0] // block_shape[0], shape[1] // block_shape[1]))

    def blocks_count(self):
        return self.blocks_count_(self.shape, self.block_shape)

    def to(self, device):
        for a in ["block_mask", "data", "cols_a", "row_start_ends_a", "rows_b", "col_start_ends_b"]:
            getattr(self, a).to(device)

    def build_indices_(self, nnz,  transpose_indices):
        nnz = nnz.transpose(0,1)

        X, Y = self.blocks_count_(self.shape, self.block_shape)

        rows = nnz[0]
        cols = nnz[1]

        block_shuffle = torch.arange(0, cols.shape[0], device = self.device)

        if transpose_indices:
            block_indices = torch.zeros(X*Y, dtype=torch.long, device = self.device)
            positions = rows * Y + cols
            block_indices[positions] = block_shuffle + 1
            block_indices = block_indices.reshape(X, Y).t().reshape(X * Y)
            block_shuffle = block_indices[block_indices.nonzero()] - 1
            block_shuffle = block_shuffle.squeeze(-1)

            X, Y = Y, X
            rows, cols = cols, rows

        row_start_ends = torch.zeros((X + 1,), dtype=torch.long, device = self.device)

        row_start_ends.index_add_(0, rows + 1, torch.ones(size=(cols.shape[0],), dtype=torch.long, device = self.device))
        row_start_ends = row_start_ends.cumsum(0).int()

        cols = torch.stack([cols, block_shuffle], 1).int()

        return cols, row_start_ends

    def build_indices(self):
        nnz = self.block_mask.nonzero()
        self.blocks = nnz.flip(-1).flatten().to(dtype=torch.int32)
        self.cols_a, self.row_start_ends_a = self.build_indices_(nnz, False)
        self.rows_b, self.col_start_ends_b  = self.build_indices_(nnz, True)

    @classmethod
    def zeros(cls, shape, n_blocks = None, blocks = None, block_shape=(32, 32), device = "cuda"):
        assert(device.startswith("cuda"))
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
        torch.randn(ret.data.shape, out=ret.data)
        return ret

    def __repr__(self):
        return "%s(shape=%s, cols=%s, row_start_ends_a=%s, data=%s, block_shape=%s)" % (self.__class__.__name__,
                                                                               self.shape,
                                                                               self.cols_a.shape,
                                                                               self.row_start_ends_a.shape,
                                                                               self.data.shape,
                                                                               self.block_shape)

    def build_coo_block_index(self):
        # Build a tensor to store the row indices.
        # It's one element too long for the moment, we'll trim it later
        rows = torch.zeros((self.cols_a.shape[0] + 1), dtype=torch.int32, device=self.device)

        # Change self.row_start_ends_a to the right type
        row_end_prepare = self.row_start_ends_a[1:].long()

        # Add ones to the start position of each new row
        rows.index_add_(0, row_end_prepare, torch.ones(size=row_end_prepare.shape, dtype=torch.int32, device=self.device))

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
            from_sparse = self.data[i * self.block_shape[0], 0]
            from_dense = dense_version[r * self.block_shape[0], c * self.block_shape[1]]
            if from_sparse != from_dense:
                raise Exception("non matching data")

        return

    def matmul_cuda(self, dense_a):
        """Compute a.matmul(self.t()) """
        import block_sparse_native
        shape_a = dense_a.shape
        shape_b = self.shape[1], self.shape[0]

        if shape_a[1] != shape_b[0]:
            raise Exception("Invalid matrices sizes (%d, %d) x (%d, %d)" % (shape_a[0], shape_a[1], shape_b[0], shape_b[1]))

        out = torch.zeros((shape_a[0], shape_b[1]), device = dense_a.device)

        cols_a = self.cols_a[:,0].contiguous()

        assert(dense_a.is_contiguous())
        assert (self.row_start_ends_a.is_contiguous())
        assert(cols_a.is_contiguous())
        assert(self.data.is_contiguous())
        assert(out.is_contiguous())

        out2 = block_sparse_native.blocksparse_matmul_transpose_cuda(dense_a,
                                                                     self.row_start_ends_a, cols_a, self.data,
                                                                     *self.shape, *self.block_shape,
                                                                     out)

        return out2

    def reverse_matmul(self, dense_a, transpose = True):
        """Compute a.matmul(self.t()) if transposed, else a.matmul(self)"""
        import block_sparse_native
        shape_a = dense_a.shape
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

        assert(self.data.is_contiguous())
        assert(out.is_contiguous())

        assert(ptr_b.is_contiguous())
        assert(ptr_b.dtype == torch.int32)
        assert(indices_b.is_contiguous())
        assert(indices_b.dtype == torch.int32)

        assert(ptr_b.shape[0] == self.blocks_count()[dim] + 1)

        verbose = False
        verbose_stride = 8

        if transpose:
            data_b = self.data
        else:
            # TEMPORARY : move this to kernel
            data = self.data.view(-1, *block_shape)
            if verbose:
                print("data #0", data.shape, data[:,::verbose_stride,::verbose_stride])
            data = data.transpose(1,2)
            if verbose:
                print("data #1", data.shape, data[:,::verbose_stride, ::verbose_stride])
            data_b = data.reshape(-1, block_shape[1]).contiguous()
            if verbose:
                print("data #2", data.shape, data[::verbose_stride,::verbose_stride])
            
        #print(ptr, indices, data, dense_a)

        #print(f"ptr={ptr}")
        #print(f"indices={indices}")
        #print(f"data={data}")
        #print(f"dense_a={dense_a}")
        if verbose:
            print("dense_a\n", dense_a[::verbose_stride,::verbose_stride])
            print("ptr", ptr_b)
            print("indices", indices_b)
            print("transpose", indices_b)

        if not dense_a.is_contiguous():
            warnings.warn("pytorch_block_sparse.BlockSparseMatrix.transposed_reverse_matmul WARNING: DEGRADED performance, dense_a is not contiguous")
            print(dense_a.stride())

        dense_a = dense_a.contiguous()

        block_sparse_native.blocksparse_matmul_cutlass(dense_a,
                                                       ptr_b, indices_b,
                                                       data_b,
                                                       dense_a.shape[0], shape_b[1], shape_b[0],
                                                       block_shape[1], block_shape[0],
                                                       out)
        if verbose:
            print("out\n", out.shape, out[::verbose_stride, ::verbose_stride])
        return out.t()

    def matmul_with_output_sparse_support(self, dense_a, dense_b, overwrite_data = False):
        """Compute  c = a.t().mm(b) where c is sparse (we just keep the results where c is non_zero)."""
        import block_sparse_native
        shape_a = dense_a.shape
        shape_b = dense_b.shape
        shape_c = self.shape

        assert(shape_a[0] == shape_b[0])
        assert(shape_c[0] == shape_a[1])
        assert(shape_c[1] == shape_b[1])

        blocks_len = len(self.blocks) // 2

        if overwrite_data:
            data = self.data
        else:
            data = torch.zeros_like(self.data)

        if not dense_a.is_contiguous():
            warnings.warn("pytorch_block_sparse.BlockSparseMatrix.matmul_with_output_sparse_support WARNING: DEGRADED performance, dense_a is not contiguous")
            dense_a = dense_a.contiguous()
        if not dense_b.is_contiguous():
            warnings.warn("pytorch_block_sparse.BlockSparseMatrix.matmul_with_output_sparse_support WARNING: DEGRADED performance, dense_b is not contiguous")
            dense_b = dense_b.contiguous()

        block_sparse_native.blocksparse_matmul_back_cutlass(dense_a, dense_b,
                                                            shape_a[1], shape_b[1], shape_a[0],
                                                            self.block_shape[0], self.block_shape[1],
                                                            data,
                                                            self.blocks, blocks_len)

        return data

