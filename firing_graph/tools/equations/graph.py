# Global import

# Local import


def buo(sax_ob, sax_cm, firing_graph):
    """
    Update adjacency matrix from firing_graph vertices toward output vertices

    :param sax_ob: output backward signal
    :type sax_ob: scipy.sparse.spmatrix
    :param sax_cm: firing_graph forward signal memory
    :type: scipy.sparse.spmatrix
    :param firing_graph:
    :type firing_graph: deyep.firing_graph.data_structure.graph.FiringGraph
    :return: Update of adjacency matrice
    :rtype: scipy.sparse.spmatrix
    """
    sax_mask = firing_graph.O.multiply(firing_graph.Om)

    # Get strucutre update
    sax_Ou = sax_ob\
        .astype(firing_graph.Ow.dtype)\
        .dot(sax_cm)\
        .transpose()\
        .multiply(sax_mask)

    sax_track = (sax_ob != 0)\
        .astype(firing_graph.backward_firing['o'].dtype)\
        .dot(sax_cm)\
        .transpose()\
        .multiply(sax_mask.astype(firing_graph.backward_firing['o'].dtype))

    firing_graph.matrices['Ow'] += sax_Ou

    return sax_track


def buc(sax_cb, sax_cm, firing_graph):
    """
    Update adjacency matrix from firing_graph vertices toward firing_graph vertices

    :param sax_cb: firing_graph backward signal
    :type sax_cb: scipy.sparse.spmatrix
    :param sax_cm: firing_graph forward signal memory
    :type: sax_cm scipy.sparse.spmatrix
    :param firing_graph:
    :type firing_graph: deyep.firing_graph.data_structure.graph.FiringGraph
    :return: update of adjacency matrix
    :rtype: scipy.sparse.spmatrix
    """

    sax_mask = firing_graph.C.multiply(firing_graph.Cm)

    sax_Cu = sax_cb\
        .astype(firing_graph.Cw.dtype)\
        .dot(sax_cm)\
        .transpose()\
        .multiply(sax_mask)

    sax_track = (sax_cb != 0)\
        .astype(firing_graph.backward_firing['c'].dtype)\
        .dot(sax_cm.astype(firing_graph.backward_firing['c'].dtype))\
        .transpose()\
        .multiply(sax_mask.astype(firing_graph.backward_firing['c'].dtype))

    firing_graph.matrices['Cw'] += sax_Cu

    return sax_track


def bui(sax_cb, sax_im, firing_graph):
    """
    Update adjacency matrix from input vertices toward firing_graph vertices

    :param sax_cb: firing_graph backward signal
    :type sax_cb: scipy.sparse.spmatrix
    :param sax_im: input forward signal memory
    :type: sax_im: scipy.sparse.spmatrix
    :param firing_graph:
    :type firing_graph: deyep.firing_graph.data_structure.graph.FiringGraph
    :return: update of adjacency matrix
    :rtype: scipy.sparse.spmatrix
    """

    sax_mask = firing_graph.I.multiply(firing_graph.Im)

    sax_Iu = sax_cb\
        .astype(firing_graph.Iw.dtype)\
        .dot(sax_im)\
        .transpose()\
        .multiply(sax_mask)

    sax_track = (sax_cb != 0)\
        .astype(firing_graph.backward_firing['i'].dtype)\
        .dot(sax_im)\
        .transpose()\
        .multiply(sax_mask.astype(firing_graph.backward_firing['i'].dtype))

    firing_graph.matrices['Iw'] += sax_Iu

    return sax_track
