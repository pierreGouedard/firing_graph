# Global import
from scipy.sparse import csr_matrix, vstack, hstack, lil_matrix, csc_matrix
from numpy import int32
from multiprocessing import Pool

# Local import


def init_ftc_worker(sax_i, sax_I, sax_c, sax_C):
    # declare scope of a new global variable
    global sax_shared_i
    global sax_shared_c
    global sax_shared_I
    global sax_shared_C

    # store argument in the global variable for this process
    sax_shared_i = sax_i.tocsc() if sax_i.nnz > 0 and sax_I.nnz > 0 else None
    sax_shared_I = sax_I.astype(int32).tocsc() if sax_i.nnz > 0 and sax_I.nnz > 0 else None
    sax_shared_c = sax_c.tocsc() if sax_c.nnz > 0 and sax_C.nnz > 0 else None
    sax_shared_C = sax_C.astype(int32).tocsc() if sax_c.nnz > 0 and sax_C.nnz > 0 else None


def ftc_task(k, level):
    if level == 0:
        return csc_matrix((sax_shared_i.shape[0], 1), dtype=bool)
    elif sax_shared_c is None:
        return sax_shared_i.dot(sax_shared_I[:, k]) >= level
    elif sax_shared_i is None:
        return sax_shared_c.dot(sax_shared_C[:, k]) >= level
    else:
        return sax_shared_i.dot(sax_shared_I[:, k]) + sax_shared_c.dot(sax_shared_C[:, k]) >= level


def ftc(sax_I, sax_i, sax_C, sax_c, ax_levels, njobs=1):
    if (sax_c.nnz + sax_i.nnz == 0) or (sax_C.nnz + sax_I.nnz == 0):
        return sax_c

    with Pool(njobs, initializer=init_ftc_worker, initargs=(sax_i, sax_I, sax_c, sax_C)) as p:
        sax_c = hstack(p.starmap(ftc_task, list(enumerate(ax_levels))), format='csr', dtype=bool)

    return sax_c


def init_fast_partitioned_ftc_worker(l_partitioned_I, sax_i, ax_levels):
    # declare scope of a new global variable
    global sax_shared_i
    global l_shared_partitioned_I
    global ax_shared_levels

    # store argument in the global variable for this process
    sax_shared_i = sax_i.tocsc()
    l_shared_partitioned_I = l_partitioned_I
    ax_shared_levels = ax_levels


def fast_partitioned_ftc_task(k):
    # Init result
    sax_res = lil_matrix((1, len(l_shared_partitioned_I)), dtype=bool)
    for i, l_partitions in enumerate(l_shared_partitioned_I):
        if ax_shared_levels[i] == 0:
            continue

        stop, n, tol = False, 0, len(l_partitions) - ax_shared_levels[i]
        for s, e, sax_sub_I in l_partitions:
            if sax_shared_i[k, s:e].dot(sax_sub_I).nnz == 0:
                n += 1
                if n > tol:
                    stop = True
                    break

        if not stop:
            sax_res[0, i] = True

    return sax_res.tocsr()


def fast_partitioned_ftc(l_partitioned_I, sax_i, ax_levels, njobs=1):
    # Transmit input
    with Pool(njobs, initializer=init_fast_partitioned_ftc_worker, initargs=(l_partitioned_I, sax_i, ax_levels)) as p:
        sax_c = vstack(p.map(fast_partitioned_ftc_task, list(range(sax_i.shape[0]))), format='csr', dtype=bool)

    return sax_c


def init_fpo_worker(sax_o, sax_got):
    # declare scope of a new global variable
    global sax_shared_o
    global sax_shared_got

    # store argument in the global variable for this process
    sax_shared_o = sax_o.tocsc()
    sax_shared_got = sax_got.tocsc()


def fpo_task(i, p, r):
    return sax_shared_got[:, i].multiply(sax_shared_o[:, i]) * (r + p) - (sax_shared_o[:, i] * p)


def fpo(sax_o, sax_got, ax_p, ax_r, njobs=1):
    # compute feedback
    with Pool(njobs, initializer=init_fpo_worker, initargs=(sax_o, sax_got)) as p:
        sax_cb = hstack(p.starmap(fpo_task, [(i, p, r) for i, (p, r) in enumerate(zip(*[ax_p, ax_r]))]), dtype=int32)

    return sax_cb.tocsr()


def init_fto_worker(sax_c, sax_O):
    # declare scope of a new global variable
    global sax_shared_c
    global sax_shared_O

    # store argument in the global variable for this process
    sax_shared_c = sax_c
    sax_shared_O = sax_O


def fto_task(k):
    return sax_shared_c[k, :].dot(sax_shared_O)


def fto(sax_O, sax_c, njobs=1):
    if njobs == 1:
        return sax_c.dot(sax_O)

    # Transmit input
    with Pool(njobs, initializer=init_fto_worker, initargs=(sax_c, sax_O)) as p:
        sax_o = vstack(p.map(fto_task, list(range(sax_c.shape[0]))), format='csr', dtype=bool)

    return sax_o
