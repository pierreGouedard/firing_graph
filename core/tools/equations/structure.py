# Global import

# Local import


def buo(sax_ob, sax_cm, firing_graph):
    """
    Update adjacency matrix from core vertices toward output vertices

    :param sax_ob: output backward signal
    :type sax_ob: scipy.sparse.spmatrix
    :param sax_cm: core forward signal memory
    :type: scipy.sparse.spmatrix
    :param firing_graph:
    :type firing_graph: deyep.core.firing_graph.graph.FiringGraph
    :return: Update of adjacency matrice
    :rtype: scipy.sparse.spmatrix
    """
    # Get strucutre update
    sax_Ou = sax_ob.dot(sax_cm).transpose().multiply(firing_graph.O.astype(int))
    sax_track = (sax_ob != 0).astype(int).dot(sax_cm).transpose().multiply(firing_graph.O.multiply(firing_graph.Om).astype(int))
    firing_graph.matrices['Ow'] += sax_Ou.multiply(firing_graph.Om)

    return sax_track.multiply(firing_graph.O.multiply(firing_graph.Om).astype(int))


def buc(sax_cb, sax_cm, firing_graph):
    """
    Update adjacency matrix from core vertices toward core vertices

    :param sax_cb: core backward signal
    :type sax_cb: scipy.sparse.spmatrix
    :param sax_cm: core forward signal memory
    :type: sax_cm scipy.sparse.spmatrix
    :param firing_graph:
    :type firing_graph: deyep.core.firing_graph.graph.FiringGraph
    :return: update of adjacency matrix
    :rtype: scipy.sparse.spmatrix
    """
    sax_Cu = sax_cb.dot(sax_cm).transpose().multiply(firing_graph.C.astype(int))
    sax_track = (sax_cb != 0).astype(int).dot(sax_cm).transpose().multiply(firing_graph.C.multiply(firing_graph.Cm).astype(int))
    firing_graph.matrices['Cw'] += sax_Cu.multiply(firing_graph.Cm)

    return sax_track


def bui(sax_cb, sax_im, firing_graph):
    """
    Update adjacency matrix from input vertices toward core vertices

    :param sax_cb: core backward signal
    :type sax_cb: scipy.sparse.spmatrix
    :param sax_im: input forward signal memory
    :type: sax_im: scipy.sparse.spmatrix
    :param firing_graph:
    :type firing_graph: deyep.core.firing_graph.graph.FiringGraph
    :return: update of adjacency matrix
    :rtype: scipy.sparse.spmatrix
    """
    sax_Iu = sax_cb.dot(sax_im).transpose().multiply(firing_graph.I.astype(int))
    sax_track = (sax_cb != 0).astype(int).dot(sax_im).transpose().multiply(firing_graph.I.multiply(firing_graph.Im).astype(int))
    firing_graph.matrices['Iw'] += sax_Iu.multiply(firing_graph.Im)

    return sax_track