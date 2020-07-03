import torch
import numpy
import block_sparse_cuda

class BlockSparseMatrix:
    # cols is a list of nonzero block column indexes (int32)
    # row_start is a index into cols (int32)
    # Data is (len(cols), block_shape, block_shape)
    def __init__(self, shape, block_mask, data, block_shape=(16, 16)):
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

    @staticmethod
    def block_shape_(shape, block_shape):
        return torch.Size((shape[0] // block_shape[0], shape[1] // block_shape[1]))

    def build_indices_(self, nnz,  transpose_indices):
        nnz = nnz.transpose(0,1)
        X, Y = self.block_shape_(self.shape, self.block_shape)

        rows = nnz[0]
        cols = nnz[1]

        block_shuffle = torch.arange(0, cols.shape[0])

        if transpose_indices:
            block_indices = torch.zeros(X*Y, dtype=torch.long)
            positions = rows * Y + cols
            block_indices[positions] = block_shuffle + 1
            block_indices = block_indices.reshape(X, Y).t().reshape(X * Y)
            block_shuffle = block_indices[block_indices.nonzero()] - 1
            block_shuffle = block_shuffle.squeeze(-1)

            X, Y = Y, X
            rows, cols = cols, rows

        row_ends = torch.zeros((X,), dtype=torch.long)

        row_ends.index_add_(0, rows, torch.ones(size=(cols.shape[0],), dtype=torch.long))
        row_ends = row_ends.cumsum(0).int()

        cols = torch.stack([cols, block_shuffle], 1).int()

        return cols, row_ends

    def build_indices(self):
        nnz = self.block_mask.nonzero()
        self.cols_a, self.row_ends_a = self.build_indices_(nnz, False)
        self.rows_b, self.col_ends_b  = self.build_indices_(nnz, True)

    @classmethod
    def rand(cls, shape, n_blocks, block_shape=(16, 16)):
        if len(shape) != 2 or shape[0] % 16 != 0 or shape[1] % 16 != 0:
            raise Exception("shape should be a tuple of 2 multiples of 16")

        X, Y = cls.block_shape_(shape, block_shape)

        if n_blocks > X * Y:
            raise Exception("Too many blocks : %d > %d * %d = %d" % (n_blocks, X, Y, X * Y))
        positions = numpy.random.choice(X*Y, size=n_blocks, replace=False)
        positions = torch.tensor(positions, dtype=torch.int64).sort()[0]

        block_mask = torch.zeros(X * Y, dtype=torch.bool)
        block_mask[positions] = True
        block_mask = block_mask.view(X, Y)

        data = torch.normal(0,1.0, size = (n_blocks * block_shape[0], block_shape[1]), dtype=torch.float)

        return cls(shape, block_mask, data, block_shape)

    def __repr__(self):
        return "%s(shape=%s, cols=%s, row_ends_a=%s, data=%s, block_shape=%s)" % (self.__class__.__name__,
                                                                               self.shape,
                                                                               self.cols_a.shape,
                                                                               self.row_ends_a.shape,
                                                                               self.data.shape,
                                                                               self.block_shape)

    def build_coo_block_index(self):
        # Build a tensor to store the row indices.
        # It's one element too long for the moment, we'll trim it later
        rows = torch.zeros((self.cols_a.shape[0] + 1), dtype=torch.int32)

        # Change self.row_ends_a to the right type
        row_end_prepare = self.row_ends_a.long()

        # Add ones to the start position of each new row
        rows.index_add_(0, row_end_prepare, torch.ones(size=row_end_prepare.shape, dtype=torch.int32))

        # Accumulate those start positions to fill the remaining positions
        rows = rows.cumsum(0).int()

        # Trim the last element: it's just a left over
        rows = rows[:-1]

        # Build the coo indexes
        return torch.stack([rows, self.cols_a[:,0]], 0)

    def to_sparse(self):
        coo = self.build_coo_block_index().long()

        data = self.data.reshape(-1, *self.block_shape)

        out = torch.sparse.FloatTensor(coo, data, (self.shape[0] // self.block_shape[0], self.shape[1] // self.block_shape[1]) + self.block_shape)

        return out

    def to_dense(self):
        out = self.to_sparse()
        out = out.to_dense()
        out = out.transpose(1,2)
        out = out.reshape(self.shape[0], self.shape[1])

        return out

    def sanity_check(self):
        cols = self.cols_a
        row_end = self.row_ends_a
        shape = self.shape
        block_shape = self.block_shape

        if len(cols.shape) != 2:
            raise Exception("cols should be bidimensional, not of shape %s" % cols.shape)
        if cols.dtype != torch.int32:
            raise Exception("cols should be int32, not of type %s" % cols.dtype)
        max_col = cols[:,0].max()
        if max_col > shape[1] / block_shape[1]:
            raise Exception("cols max element (%d) cannot be larger than shape[1]/block_shape[1] (%d)" % (max_col, shape[1] / block_shape[1]))

        self.cols = cols

        if len(row_end.shape) != 1:
            raise Exception("row_end should be unidimensional, not of shape %s" % row_end.shape)
        if row_end.shape[0] != shape[0] / block_shape[0]:
            raise Exception("row_end.shape[0] (%d) should be equal to shape[0]/block_shape[0] (%d)" % (row_end.shape[0], shape[0] / block_shape[0]))
        if row_end.dtype != torch.int32:
            raise Exception("row_end should be int32, not of type %s" % row_end.dtype)

        max_row_end = row_end.max()
        if max_row_end > self.cols.shape[0]:
            raise Exception("row_end max element (%d) cannot be larger than cols count (%d)" % (max_row_end, self.cols.shape[0]))
        last_row_end = row_end[-1]
        if last_row_end != self.cols.shape[0]:
            raise Exception("row_end last element (%d) should be equal to cols count (%d)" % (last_row_end, self.cols.shape[0]))
        self.row_end = row_end


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

    def transposed_matmul(self, dense_a):
        """Compute a.matmul(self.t()) """

        shape_a = dense_a.shape
        shape_b = self.shape[1], self.shape[0]

        if shape_a[1] != shape_b[0]:
            raise Exception("Invalid matrices sizes (%d, %d) x (%d, %d)" % (shape_a[0], shape_a[1], shape_b[0], shape_b[1]))

        out = torch.zeros((shape_a[0], shape_b[1]), device = dense_a.device)

        cols_a = self.cols_a.flatten()

        assert(dense_a.is_contiguous())
        assert (self.row_ends_a.is_contiguous())
        assert(cols_a.is_contiguous())
        assert(self.data.is_contiguous())
        assert(out.is_contiguous())

        out2 = block_sparse_cuda.blocksparse_matmul_transpose(dense_a,
                                                              self.row_ends_a, cols_a, self.data,
                                                              *self.shape, *self.block_shape,
                                                              out)
        return out2