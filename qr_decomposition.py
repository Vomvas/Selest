"""
QR Decomposition and QR Algorithm engines
"""
from . import complex
from .library import for_range, for_range_parallel, print_ln
from .types import *
from .GC.types import sbitfix
from Compiler.mpc_math import sqrt


class QRDecomposition(object):
    """
    Performs QR decomposition on given matrix A of type complex.CplxMatrix
    """

    def __init__(self, A: complex.CplxMatrix, process="gram_schmidt"):

        self.process = process
        self.input_matrix = complex.CplxMatrix(A.rows, A.col)
        self.input_matrix.assign(A)
        self.value_type = self.input_matrix.value_type.value_type

        self.Q = None
        self.R = None

        # Anything below this threshold is considered equal to zero
        self.zero_thres = 1e-5

        assert isinstance(A, complex.CplxMatrix)

    def delete(self):
        """Delete elements from memory"""
        self.input_matrix.delete()
        self.Q.delete()
        self.R.delete()

    def run_gram_schmidt(self):
        """Performs Gram Schmidt process on input complex matrix A"""

        # GS process works on the columns of A, therefore we have to initially transpose the matrix
        mat = self.input_matrix.transpose()

        basis = complex.CplxMatrix.zeros(mat.sizes[0], mat.sizes[1])

        # Temp matrix of all past contributions
        projection_matrix = complex.CplxMatrix(basis.sizes[0], mat.sizes[1])

        # For every input vector
        @for_range(mat.sizes[0])
        def _(vct_index):
            v = mat.get_row(vct_index)

            # Initialize projection matrix to zero matrix
            projection_matrix.assign_all(complex.scplx())

            # For every vector already added in the basis calculate the projections
            if complex.scplx.value_type == sbitfix:
                @for_range(basis.sizes[0])
                def _(basis_index):
                    b = basis.get_row(basis_index)

                    # Basis vectors are normalized (norm squared = 1)
                    projection = ((b.conjugate() * v)) * b
                    projection_matrix.set_row(basis_index, projection)
            else:
                @for_range_parallel(basis.sizes[0], basis.sizes[0])
                def _(basis_index):
                    b = basis.get_row(basis_index)

                    # Basis vectors are normalized (norm squared = 1)
                    projection = ((b.conjugate() * v)) * b
                    projection_matrix.set_row(basis_index, projection)

            # Add all past contributions together
            contr_sum = complex.CplxArray(mat.sizes[1])

            @for_range(contr_sum.length)
            def _(contr_index):
                contr_sum[contr_index] = sum(projection_matrix.get_column(contr_index))

            # Determine if the new vector should be added to the basis
            w = complex.CplxArray(v.length)
            w.assign(v-contr_sum)

            w_norm = w.norm(type="l2")
            normalized_w = w / (w_norm)

            basis.set_row(vct_index, normalized_w)
        return basis.transpose()

    def gram_schmidt_r(self):
        """Computes matrix R of the Gram Schmidt process based on A and Q"""
        if self.Q is None:
            raise RuntimeError("Cannot compute matrix R if matrix Q is None")

        input_cols = self.input_matrix.transpose()
        mat_q_cols = self.Q.transpose()

        r = complex.CplxMatrix.zeros(mat_q_cols.sizes[0], mat_q_cols.sizes[1])

        if complex.scplx.value_type == sbitfix:
            @for_range(r.sizes[0])
            def _(i):
                @for_range(r.sizes[1] - i)
                def _(j):
                    r[i][r.sizes[1] - j - 1] = mat_q_cols.get_row(i).conjugate() * input_cols.get_row(j)
        else:
            @for_range_parallel(r.sizes[0], r.sizes[0])
            def _(i):
                @for_range_parallel(r.sizes[1] - i, r.sizes[1] - i)
                def _(j):
                    r[i][r.sizes[1] - j - 1] = mat_q_cols.get_row(i).conjugate() * input_cols.get_row(j)
        return r

    def execute(self):
        """
        Starts the QR decomposition of input complex matrix A
        :return: Q, R complex matrices that satisfy A = Q*R where Q is unitary and R is upper diagonal
        """
        if self.process == "gram_schmidt":
            self.Q = self.run_gram_schmidt()
            self.R = self.gram_schmidt_r()
            return self.Q, self.R
        else:
            return NotImplemented


class QRAlgorithm(object):
    """
    QR Algorithm engine
    """

    def __init__(self, A: complex.CplxMatrix, iterations=1, deflate=False, deflate_iter=3,
                 decomp_process="gram_schmidt", compute_transformation_matrix=True):

        self.input_matrix = A
        self.iterations = iterations
        self.compute_transformation_matrix = compute_transformation_matrix
        self.deflate = deflate
        self.deflate_iter = deflate_iter
        self.decomp_process = decomp_process

    def run(self, debug=False):
        """Execute QR Algorithm"""
        A = self.input_matrix
        Q = complex.CplxMatrix(A.rows, A.col)
        R = complex.CplxMatrix(A.rows, A.col)

        if self.compute_transformation_matrix:
            Q_composite = complex.CplxMatrix.eye(A.rows)
        else:
            Q_composite = None
        @for_range(self.iterations)
        def _(it):

            if debug:
                print_ln("Starting iter, A:")
                A.print_reveal()

            qr_dec = QRDecomposition(A, process=self.decomp_process)
            qr_dec.execute()
            Q.assign(qr_dec.Q)
            R.assign(qr_dec.R)

            if debug:
                print_ln("Q matrix:")
                Q.print_reveal()
                print_ln("R matrix:")
                R.print_reveal()
                print_ln("R*Q:")
                (R * Q).print_reveal()

            A.assign(R * Q)
            if self.compute_transformation_matrix:
                Q_composite.assign(Q_composite * Q)
            if debug:
                print_ln("Similar matrix A_sim:")
                A.print_reveal()
        return A, Q_composite
