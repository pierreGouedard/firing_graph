# Global imports
import numpy as np
from scipy.sparse import diags, lil_matrix, hstack, vstack


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


def create_empty_matrices(n_inputs, n_outputs, n_core):

    return {
        'Im': lil_matrix((n_inputs, n_core)),
        'Cm': lil_matrix((n_core, n_core)),
        'Om': lil_matrix((n_core, n_outputs)),
        'Iw': lil_matrix((n_inputs, n_core)),
        'Cw': lil_matrix((n_core, n_core)),
        'Ow': lil_matrix((n_core, n_outputs))
    }


def augment_matrices(d_matrices_a, d_matrices_b):

    assert d_matrices_a['Iw'].shape[0] == d_matrices_b['Iw'].shape[0], "shape of inputs matrices doesn't match"
    assert d_matrices_a['Ow'].shape[1] == d_matrices_b['Ow'].shape[0], "shape of outputs matrices doesn't match"

    # Merge io matrices
    d_matrices_a['Iw'] = hstack([d_matrices_a['Iw'], d_matrices_b['Iw']])
    d_matrices_a['Im'] = hstack([d_matrices_a['Im'], d_matrices_b['Im']])
    d_matrices_a['Ow'] = vstack([d_matrices_a['Ow'], d_matrices_b['Ow']])
    d_matrices_a['Om'] = vstack([d_matrices_a['Om'], d_matrices_b['Om']])

    # Merge Core matrices
    sax_Cw_upper = hstack([d_matrices_a['Cw'], lil_matrix((d_matrices_a['Cw'].shape[0], d_matrices_b['Cw'].shape[1]))])
    sax_Cw_lower = hstack([lil_matrix((d_matrices_b['Cw'].shape[0], d_matrices_a['Cw'].shape[0])), d_matrices_b['Cw']])
    d_matrices_a['Cw'] = vstack([sax_Cw_upper, sax_Cw_lower])

    sax_Cm_upper = hstack([d_matrices_a['Cm'], lil_matrix((d_matrices_a['Cm'].shape[0], d_matrices_b['Cm'].shape[1]))])
    sax_Cm_lower = hstack([lil_matrix((d_matrices_b['Cm'].shape[0], d_matrices_a['Cm'].shape[0])), d_matrices_b['Cm']])
    d_matrices_a['Cm'] = vstack([sax_Cm_upper, sax_Cm_lower])

    return d_matrices_a


def add_core_vertices(d_matrices, n_core, offset):

    # Get I/O dimensions
    n_inputs, n_outputs = d_matrices['Iw'].shape[0], d_matrices['Ow'].shape[1]

    # Update input matrices
    d_matrices['Iw'] = hstack([d_matrices['Iw'][:, :offset], lil_matrix((n_inputs, n_core))])
    d_matrices['Im'] = hstack([d_matrices['Im'][:, :offset], lil_matrix((n_inputs, n_core))])

    # Update core matrices
    d_matrices['Cw'] = hstack([d_matrices['Cw'][:, :offset], lil_matrix((n_core, n_core))])
    d_matrices['Cw'] = vstack([d_matrices['Cw'][:, :offset], lil_matrix((n_core, d_matrices['Cw'].shape[1]))])
    d_matrices['Cm'] = hstack([d_matrices['Cm'][:, :offset], lil_matrix((n_core, n_core))])
    d_matrices['Cm'] = vstack([d_matrices['Cm'][:, :offset], lil_matrix((n_core, d_matrices['Cw'].shape[1]))])

    # Update output matrices
    d_matrices['Ow'] = vstack([d_matrices['Ow'][:, :offset], lil_matrix((n_core, n_outputs))])
    d_matrices['Om'] = vstack([d_matrices['Om'][:, :offset], lil_matrix((n_core, n_outputs))])

    return d_matrices
