# Global import
from scipy.sparse import csc_matrix, csr_matrix, vstack, diags
from numpy import newaxis

# Local import


def fti(server, firing_graph, batch_size):
    """
    Initialise the forward signal of input vertice with fresh input

    :param server:
    :type server: firing_graph.imputers.ArrayServer
    :param firing_graph:
    :type firing_graph: deyep.firing_graph.data_structure.graph.FiringGraph
    :param: batch_size: size of forward batch
    :type: int
    :return: Forward signal of input vertice
    :rtype: scipy.sparse.spmatrix
    """
    sax_i = server.next_forward(n=batch_size).sax_data_forward

    return sax_i


def ftc(sax_C, sax_I, sax_c, sax_i):
    """
    Transmit signal through firing_graph vertices of firing graph

    :param sax_C: Matrix of direct connection of firing_graph vertices
    :type sax_C: scipy.sparse.spmatrix
    :param sax_I: Matrices of direct connection of input vertices toward firing_graph vertices
    :type sax_I: scipy.sparse.spmatrix
    :param sax_c: Forward signal of firing_graph vertices
    :type sax_c: scipy.sparse.spmatrix
    :param sax_i: Forward signal of input vertices
    :type sax_i: scipy.sparse.spmatrix
    :return: scipy.sparse forward signal of firing_graph vertices
    :rtype: scipy.sparse.spmatrix
    """
    sax_c = sax_i.dot(sax_I) + sax_c.dot(sax_C)

    return sax_c


def fto(sax_O, sax_c):
    """
    Transmit signal of firing_graph vertices to output vertices of firing graph

    :param sax_O: Matrix of direct connection of firing_graph vertices toward output vertices
    :type sax_O: scipy.sparse.spmatrix
    :param sax_c: Forward signal of firing_graph vertices
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
    sax_im = vstack([
        csr_matrix(sax_i.shape, dtype=sax_i.dtype),
        sax_i,
        sax_im[sax_i.shape[0]:sax_im.shape[0] - sax_i.shape[0], :]
    ])

    return sax_im


def fpc(sax_c, sax_cm, ax_levels):
    """
    Compute new firing_graph forward signal  and store into firing_graph forward memory signal if it is not None

    :param sax_c: received forward signal
    :type sax_c: scipy.sparse.spmatrix
    :param sax_cm: forward memory of firing_graph vertice or None
    :type sax_cm: scipy.sparse.spmatrix or None
    :param ax_levels: levels of forward vertex
    :type ax_levels: numpy.array
    :return: new forward memory
    :rtype: scipy.sparse.spmatrix
    """

    # Compare forward strength to level for activation
    sax_c =  (sax_c > (sax_c > 0).dot(diags((ax_levels - 1).clip(min=0), format='csc'))).astype(sax_c.dtype)

    # Get forward signal
    if sax_cm is not None:
        sax_cm = vstack([sax_c, sax_cm[:sax_cm.shape[0] - sax_c.shape[0], :]])
        return sax_c, sax_cm

    return sax_c


def fpo(sax_o, server, batch_size, ax_p, ax_q):
    """
    Compute feedback from output forward signal and ground of truth of activation

    :param sax_o: received forward signals
    :type sax_o: scipy.sparse.spmatrix
    :param server:
    :type server: firing_graph.tools.helpers.server.ArrayServer
    :param batch_size:
    :type batch_size: int
    :param ax_p:
    :type: int
    :param ax_q:
    :type: int
    :return: Feedback
    :rtype: scipy.sparse.spmatrix
    """

    # Get got signal
    sax_got = server.next_backward(n=batch_size).sax_data_backward
    sax_o = (sax_o > 0).astype(server.dtype_backward)

    # update output with mask if any
    if server.sax_mask_forward is not None:
        sax_o += server.sax_mask_forward.multiply(sax_o)
        sax_o.data %= 2
        sax_o.eliminate_zeros()

    # Compute feedback signal
    sax_ob = sax_got.tocsc().multiply(sax_o).dot(diags(ax_p + ax_q))
    sax_ob -= sax_o.dot(diags(ax_p))

    return sax_ob.transpose()
