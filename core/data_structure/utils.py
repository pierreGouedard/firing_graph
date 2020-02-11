# Global imports
import numpy as np
from scipy.sparse import diags, lil_matrix, csc_matrix, csr_matrix, hstack, vstack


def mat_from_tuples(ni, no, nc, l_edges, weights):
    """
    Take list of tuple (str x, str y), dimension of network and weights and set matrices of network
    :param l_edges: [(str x, str y)] contains name of vertex class in ('input', 'core', 'output') and their index
    :param ni: int number of input vertex
    :param nc: int number of core vertex
    :param no: int number of core vertex
    :param l_weights: either not set of int weights of every edges or list of weight (size equal to size of l_edges
    :return: 3 sparse matrices of the network
    """
    # Init matrices
    sax_in = lil_matrix(np.zeros([ni, nc]))
    sax_core = lil_matrix(np.zeros([nc, nc]))
    sax_out = lil_matrix(np.zeros([nc, no]))

    i = 0
    for (_n, n_) in l_edges:
        if 'input' in _n:
            if isinstance(weights, int):
                v = weights
            elif isinstance(weights, list):
                try:
                    v = weights[i]
                except IndexError:
                    raise ValueError("Dimension of weights does not match dimension of list of edges")
            else:
                raise ValueError("Value of the weights not correct")

            sax_in[int(_n.split('_')[1]), int(n_.split('_')[1])] = v

        elif 'core' in _n:
            if 'core' in n_:
                if isinstance(weights, int):
                    v = weights
                elif isinstance(weights, list):
                    try:
                        v = weights[i]
                    except IndexError:
                        ValueError("Dimension of weights does not match dimension of list of edges")
                else:
                    raise ValueError("Value of the weights not correct")

                sax_core[int(_n.split('_')[1]), int(n_.split('_')[1])] = v

            elif 'output' in n_:
                if isinstance(weights, int):
                    v = weights
                elif isinstance(weights, list):
                    try:
                        v = weights[i]
                    except IndexError:
                        ValueError("Dimension of weights does not match dimension of list of edges")
                else:
                    raise ValueError("Value of the weights not correct")

                sax_out[int(_n.split('_')[1]), int(n_.split('_')[1])] = v

        i += 1

    return sax_in.tocsc(), sax_core.tocsc(), sax_out.tocsc()


def mat_mask_from_vertice_mask(sax_I, sax_C, sax_O, mask_vertice_drain):
    """

    :param sax_I:
    :param sax_C:
    :param sax_O:
    :param mask_vertice_drain:
    :return:
    """
    mask_mat = {}
    mask_mat['Im'] = diags(mask_vertice_drain['I']).dot(sax_I > 0)
    mask_mat['Cm'] = diags(mask_vertice_drain['C']).dot(sax_C > 0)
    mask_mat['Om'] = diags(mask_vertice_drain['C']).dot(sax_O > 0)

    return mask_mat


def gather_matrices(ax_in, ax_core, ax_out):
    """
    from numpy array of direct link between different king of vertices of firing graph return the global matrices of
    direct link (no distinction of vertex type)

    :param ax_in: numpy.array of direct link from input vertices toward core vertices
    :param ax_core: numpy.array of direct link of vertices
    :param ax_out: numpy.array of direct link from core vertices toward output vertices
    :return: numpy.array of direct link of vertices of firing graph
    """

    ax_fg = np.vstack((ax_in, ax_core))

    ax_fg = np.hstack((np.zeros((ax_fg.shape[0], ax_in.shape[0])), ax_fg))

    ax_out_ = np.vstack((np.zeros((ax_in.shape[0], ax_out.shape[1])), ax_out))

    ax_fg = np.hstack((ax_fg,  ax_out_))

    ax_fg = np.vstack((ax_fg, np.zeros((ax_out.shape[1], ax_fg.shape[1]))))

    return ax_fg


def set_matrices_spec(d_matrices, write_mode=True):
    set_matrices_type(d_matrices)
    set_matrices_format(d_matrices, write_mode)


def set_matrices_type(d_matrices):

    d_matrices.update({
        'Iw': d_matrices['Iw'].astype(np.int32),
        'Cw': d_matrices['Cw'].astype(np.int32),
        'Ow': d_matrices['Ow'].astype(np.int32),
        'Im': d_matrices['Im'].astype(bool),
        'Cm': d_matrices['Cm'].astype(bool),
        'Om': d_matrices['Om'].astype(bool),
    })


def set_matrices_format(d_matrices, write_mode=True):

    d_matrices.update({
        'Im': d_matrices['Im'].tolil(),
        'Cm': d_matrices['Cm'].tolil(),
        'Om': d_matrices['Om'].tolil(),
    })

    if write_mode:
        d_matrices.update({
            'Iw': d_matrices['Iw'].tolil(),
            'Cw': d_matrices['Cw'].tolil(),
            'Ow': d_matrices['Ow'].tolil(),
        })

    else:
        d_matrices.update({
            'Iw': d_matrices['Iw'].tocsc(),
            'Cw': d_matrices['Cw'].tocsc(),
            'Ow': d_matrices['Ow'].tocsc(),
        })


def reduce_backward_firing(d_backward_firing, l_indices):
    return {
        'i': d_backward_firing['i'][:, l_indices], 'o': d_backward_firing['o'][l_indices, :],
        'c': d_backward_firing['c'][l_indices, :][:, l_indices]
    }


def create_empty_backward_firing(n_inputs, n_outputs, n_core, dtype=np.uint32):
    return {
        'i': csc_matrix((n_inputs, n_core), dtype=dtype),
        'c': csc_matrix((n_core, n_core), dtype=dtype),
        'o': csc_matrix((n_core, n_outputs), dtype=dtype),
    }


def create_empty_matrices(n_inputs, n_outputs, n_core, write_mode=True):

    d_matrices = {
            'Im': lil_matrix((n_inputs, n_core), dtype=bool),
            'Cm': lil_matrix((n_core, n_core), dtype=bool),
            'Om': lil_matrix((n_core, n_outputs), dtype=bool)
    }

    if write_mode:
        d_matrices.update({
            'Iw': lil_matrix((n_inputs, n_core)),
            'Cw': lil_matrix((n_core, n_core)),
            'Ow': lil_matrix((n_core, n_outputs))
        })

    else:
        d_matrices.update({
            'Iw': csc_matrix((n_inputs, n_core)),
            'Cw': csc_matrix((n_core, n_core)),
            'Ow': csc_matrix((n_core, n_outputs))
        })

    return d_matrices


def reduce_matrices(d_matrices, l_indices):

    return {
        'Iw': d_matrices['Iw'][:, l_indices], 'Im': d_matrices['Im'][:, l_indices],
        'Ow': d_matrices['Ow'][l_indices, :], 'Om': d_matrices['Om'][l_indices, :],
        'Cw': d_matrices['Cw'][l_indices, :][:, l_indices], 'Cm': d_matrices['Cm'][l_indices, :][:, l_indices]
    }


def augment_matrices(d_matrices_a, d_matrices_b, write_mode=True):

    # make sure dimension match
    assert d_matrices_a['Iw'].shape[0] == d_matrices_b['Iw'].shape[0], "shape of inputs matrices doesn't match"
    assert d_matrices_a['Ow'].shape[1] == d_matrices_b['Ow'].shape[1], "shape of outputs matrices doesn't match"

    # Merge Core matrices
    sax_Cw_upper = hstack([d_matrices_a['Cw'], csc_matrix((d_matrices_a['Cw'].shape[0], d_matrices_b['Cw'].shape[1]))])
    sax_Cw_lower = hstack([csc_matrix((d_matrices_b['Cw'].shape[0], d_matrices_a['Cw'].shape[0])), d_matrices_b['Cw']])

    sax_Cm_upper = hstack([d_matrices_a['Cm'], csc_matrix((d_matrices_a['Cm'].shape[0], d_matrices_b['Cm'].shape[1]))])
    sax_Cm_lower = hstack([csc_matrix((d_matrices_b['Cm'].shape[0], d_matrices_a['Cm'].shape[0])), d_matrices_b['Cm']])

    d_matrices = {
        'Im': hstack([d_matrices_a['Im'], d_matrices_b['Im']]),
        'Om': vstack([d_matrices_a['Om'], d_matrices_b['Om']]),
        'Cm': vstack([sax_Cm_upper, sax_Cm_lower]),
        'Iw': hstack([d_matrices_a['Iw'], d_matrices_b['Iw']]),
        'Ow': vstack([d_matrices_a['Ow'], d_matrices_b['Ow']]),
        'Cw': vstack([sax_Cw_upper, sax_Cw_lower]),
    }

    if write_mode is not None:
        set_matrices_format(d_matrices, write_mode)

    return d_matrices


def add_core_vertices(d_matrices, n_core, offset, write_mode=True):

    # Get I/O dimensions
    n_inputs, n_outputs = d_matrices['Iw'].shape[0], d_matrices['Ow'].shape[1]

    # Update core matrices
    sax_Cw_upper= hstack([d_matrices['Cw'][:offset, :offset], csc_matrix((offset, n_core))])
    sax_Cm_upper = hstack([d_matrices['Cm'][:offset, :offset], csc_matrix((offset, n_core))])

    d_matrices = {
        'Iw': hstack([d_matrices['Iw'][:, :offset], csc_matrix((n_inputs, n_core))]),
        'Im': hstack([d_matrices['Im'][:, :offset], csc_matrix((n_inputs, n_core))]),
        'Ow': vstack([d_matrices['Ow'][:, :offset], csr_matrix((n_core, n_outputs))]),
        'Om': vstack([d_matrices['Om'][:, :offset], csr_matrix((n_core, n_outputs))]),
        'Cw': vstack([sax_Cw_upper, csr_matrix((n_core, sax_Cw_upper.shape[1]))]),
        'Cm': vstack([sax_Cm_upper, csr_matrix((n_core, sax_Cm_upper.shape[1]))])
    }

    if write_mode is not None:
        set_matrices_format(d_matrices, write_mode)

    return d_matrices
