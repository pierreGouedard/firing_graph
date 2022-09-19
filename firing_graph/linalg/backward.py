# Global import
from scipy.sparse import csr_matrix, vstack, diags, lil_matrix
from numpy import int32
from multiprocessing import Pool


def init_bui_worker(sax_cb, sax_i, sax_mask):
    # declare scope of a new global variable
    global sax_shared_i
    global sax_shared_cb
    global sax_shared_mask

    # store argument in the global variable for this process
    sax_shared_i = sax_i.T.tocsc()
    sax_shared_cb = sax_cb.tocsc()
    sax_shared_mask = sax_mask.tocsc()


def bui_task(k):

    # Init result
    sax_val = lil_matrix((1, sax_shared_cb.shape[1]), dtype=int32)
    sax_cnt = lil_matrix((1, sax_shared_cb.shape[1]), dtype=int32)

    # Compute masked cb
    ax_mask = sax_shared_mask[k, :].A[0]
    sax_cb_masked = sax_shared_cb[:, ax_mask]

    # Set values
    sax_val[:, ax_mask] = sax_shared_i[k, :].dot(sax_cb_masked)
    sax_cnt[:, ax_mask] = sax_shared_i[k, :].dot((sax_cb_masked != 0).astype(int32))

    return sax_val.tocsr(), sax_cnt.tocsr()


def bui(sax_cb, sax_i, fg, njobs=1):

    if any(map(lambda x: x.nnz == 0, [sax_cb, sax_i, fg.I, fg.Im])):
        return csr_matrix(fg.I.shape)

    with Pool(njobs, initializer=init_bui_worker, initargs=(sax_cb, sax_i, fg.I.multiply(fg.Im))) as p:
        l_res = p.map(bui_task, list(range(fg.I.shape[0])))

    # Update input matrice of fg
    fg.matrices['Iw'] += vstack([x[0] for x in l_res], format='csr', dtype=int32)

    return vstack([x[1] for x in l_res], format='csr', dtype=int32)
