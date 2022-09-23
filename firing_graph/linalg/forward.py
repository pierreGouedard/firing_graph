# Global import
from scipy.sparse import csr_matrix
from numpy import int32

# Local import


def ftc(sax_I, sax_i, sax_C, sax_c, ax_levels):
    if (sax_c.nnz + sax_i.nnz == 0) or (sax_C.nnz + sax_I.nnz == 0):
        return sax_c

    elif sax_c.nnz == 0 or sax_C.nnz == 0:
        sax_prod = sax_i.astype(int32).dot(sax_I)

    elif sax_i.nnz == 0 or sax_I.nnz == 0:
        sax_prod = sax_c.astype(int32).dot(sax_C)

    else:
        sax_prod = sax_i.astype(int32).dot(sax_I) + sax_c.astype(int32).dot(sax_C)

    return sax_prod > (sax_prod > 0).multiply(csr_matrix((ax_levels - 1).clip(min=0), dtype=int32))


def fpo(sax_o, sax_got, ax_p, ax_r):
    return (
            sax_got.multiply(sax_o).multiply(csr_matrix(ax_r + ax_p, dtype=int32)) -
            (sax_o.multiply(csr_matrix(ax_p, dtype=int32)))
    )


def fto(sax_O, sax_c):
    return sax_c.dot(sax_O)
