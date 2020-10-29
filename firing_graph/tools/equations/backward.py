# Global import
from scipy.sparse import csc_matrix, hstack


def btc(sax_ob, sax_cb, sax_cm, sax_O, sax_C):
    """
    Transmit backward signal

    :param sax_ob: output backward signal
    :type sax_ob: scipy.sparse.spmatrix
    :param sax_cb: firing_graph backward signal
    :type sax_cb: scipy.sparse.spmatrix
    :param sax_O: adjacency matrix from firing_graph vertices toward output vertices
    :type sax_O: scipy.sparse.spmatrix
    :param sax_C: adjacency matrix from firing_graph vertices toward firing_graph vertices
    :type sax_C: scipy.sparse.spmatrix
    :return: backward signal
    :rtype: scipy.sparse.spmatrix
    """

    sax_cb = sax_O.dot(sax_ob) + sax_C.dot(sax_cb)
    sax_cb = sax_cb.multiply(sax_cm.transpose().astype(sax_cb.dtype))
    return sax_cb


def bpo(sax_ob, mem_size, batch_size):
    """
    Build backward signal from feedback

    :param sax_ob: feedback signal
    :type sax_ob: scipy.sparse.spmatrix
    :param mem_size:
    :type int
    :param batch_size:
    :type int
    :return: backward signal
    :rtype scipy.sparse.spmatrix
    """
    sax_ob = hstack([
        csc_matrix((sax_ob.shape[0], batch_size), dtype=sax_ob.dtype),
        sax_ob,
        csc_matrix((sax_ob.shape[0], mem_size - 2 * batch_size), dtype=sax_ob.dtype)
    ])

    return sax_ob


def bpc(sax_cb, batch_size):
    """
    Offset backward signal

    :param sax_cb: backward signal
    :type sax_cb: scipy.sparse.spmatrix
    :param batch_size:
    :type batch_size: scipy.sparse.spmatrix
    :return: offset backward signal
    :rtype: scipy.sparse.spmatrix
    """
    sax_cb = hstack([
        csc_matrix((sax_cb.shape[0], 2 * batch_size), dtype=sax_cb.dtype),
        sax_cb[:, :sax_cb.shape[1] - 2 * batch_size]
    ])
    return sax_cb
