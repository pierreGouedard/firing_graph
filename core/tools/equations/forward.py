# Global import
from scipy.sparse import csc_matrix, csr_matrix, vstack
from numpy import newaxis

# Local import


def fti(imputer, firing_graph, batch_size):
    """
    Initialise the forward signal of input vertice with fresh input

    :param imputer:
    :type imputer: deyep.core.imputers.comon.Imputer
    :param firing_graph:
    :type firing_graph: deyep.core.data_structure.graph.FiringGraph
    :param: batch_size: size of forward batch
    :type: int
    :return: Forward signal of input vertice
    :rtype: scipy.sparse.spmatrix
    """
    sax_i = csr_matrix((0, firing_graph.I.shape[0]))
    for _ in range(batch_size):
        sax_i = vstack([sax_i, imputer.next_forward().tocsr()])

    return sax_i.astype(int)


def ftc(sax_C, sax_I, sax_c, sax_i):
    """
    Transmit signal through core vertices of firing graph

    :param sax_C: Matrix of direct connection of core vertices
    :type sax_C: scipy.sparse.spmatrix
    :param sax_I: Matrices of direct connection of input vertices toward core vertices
    :type sax_I: scipy.sparse.spmatrix
    :param sax_c: Forward signal of core vertices
    :type sax_c: scipy.sparse.spmatrix
    :param sax_i: Forward signal of input vertices
    :type sax_i: scipy.sparse.spmatrix
    :return: scipy.sparse forward signal of core vertices
    :rtype: scipy.sparse.spmatrix
    """
    sax_c = sax_i.dot(sax_I) + sax_c.dot(sax_C)

    return sax_c


def fto(sax_O, sax_c):
    """
    Transmit signal of core vertices to output vertices of firing graph

    :param sax_O: Matrix of direct connection of core vertices toward output vertices
    :type sax_O: scipy.sparse.spmatrix
    :param sax_c: Forward signal of core vertices
    :type sax_c: scipy.sparse.spmatrix
    :return: Output forward signal
    :rtype: scipy.sparse.spmatrix
    """
    sax_o = sax_c.dot(sax_O)

    return sax_o


def fpi(sax_i, sax_im):
    """
    Store new input into input forward memory

    :param sax_i: new input
    :type sax_i: scipy.sparse.spmatrix
    :param sax_im: input memory
    :type sax_im: scipy.sparse.spmatrix
    :return: new forward memory
    :rtype: scipy.sparse.spmatrix
    """
    sax_im = vstack([csr_matrix(sax_i.shape), sax_i, sax_im[sax_i.shape[0]:sax_im.shape[0] - sax_i.shape[0], :]])

    return sax_im


def fpc(sax_c, sax_cm, ax_levels):
    """
    Store new core forward signal into core forward memory

    :param sax_c: received forward signal
    :type sax_c: scipy.sparse.spmatrix
    :param sax_cm: forward memory of core vertice
    :type sax_cm: scipy.sparse.spmatrix
    :param ax_levels: levels of forward vertex
    :type ax_levels: numpy.array
    :return: new forward memory
    :rtype: scipy.sparse.spmatrix
    """
    # Compare forward strength to level for activation
    sax_levels = csc_matrix((ax_levels - 1).clip(min=0)[newaxis, :].repeat(sax_c.shape[0], axis=0))
    sax_c = (sax_c - sax_levels > 0).astype(int)

    # Get forward signal
    sax_cm = vstack([sax_c, sax_cm[:sax_cm.shape[0] - sax_c.shape[0], :]])

    return sax_c, sax_cm


def fpo(sax_o, imputer, batch_size, p, q):
    """
    Compute feedback from output forward signal and ground of truth of activation

    :param sax_o: received forward signals
    :type sax_o: scipy.sparse.spmatrix
    :param imputer: deyep.core.imputers.comon.Imputer
    :type imputer: scipy.sparse.spmatrix
    :param batch_size:
    :type batch_size: int
    :param p:
    :type: int
    :param q:
    :type: int
    :return: Feedback
    :rtype: scipy.sparse.spmatrix
    """

    # Get ground of truth
    sax_got = csr_matrix((0, sax_o.shape[1]))
    for _ in range(batch_size):
        sax_got = vstack([sax_got, imputer.next_backward().tocsr()])

    # Compute feedback
    sax_ob = ((p + q) * sax_got.multiply(sax_o > 0)) - (p * (sax_o > 0).astype(int))

    return sax_ob.transpose().tocsc()
