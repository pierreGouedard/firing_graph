# Global imports
import pickle
import random
import string
from scipy.sparse import lil_matrix, csc_matrix, hstack, vstack
import copy
from numpy import array

# Local import
from ..data_structure import utils
from core.tools.equations.forward import ftc, fto, fpc


class FiringGraph(object):
    """
    This class implement the main data structure used for fiting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """
    def __init__(self, project, ax_levels, matrices, depth=2, graph_id=None, is_drained=False, partitions=None):

        if graph_id is None:
            graph_id = ''.join([random.choice(string.ascii_letters) for _ in range(5)])

        self.project = project
        self.graph_id = graph_id
        self.is_drained = is_drained

        # Parameter
        self.depth = depth
        self.levels = ax_levels
        self.matrices = matrices
        self.partitions = partitions

        # Tracking
        self.backward_firing = {
            'i': lil_matrix(matrices['Iw'].shape), 'c': lil_matrix(matrices['Cw'].shape),
            'o': lil_matrix(matrices['Ow'].shape),
        }

        self.forward_firing = {
            'i': lil_matrix((1, matrices['Iw'].shape[0])), 'c': lil_matrix((1, matrices['Cw'].shape[0])),
            'o': lil_matrix((1, matrices['Ow'].shape[1]))
        }

    @property
    def C(self):
        return self.Cw > 0

    @property
    def Cw(self):
        return self.matrices['Cw']

    @property
    def Cm(self):
        return self.matrices['Cm']

    @property
    def O(self):
        return self.Ow > 0

    @property
    def Ow(self):
        return self.matrices['Ow']

    @property
    def Om(self):
        return self.matrices['Om']

    @property
    def I(self):
        return self.Iw > 0

    @property
    def Iw(self):
        return self.matrices['Iw']

    @property
    def Im(self):
        return self.matrices['Im']

    def update_backward_firing(self, key, sax_M):

        assert key in ['I', 'C', 'O'], "Key should be in {}".format(['I', 'C', 'O'])

        if key == 'I':
            self.backward_firing['i'] += sax_M

        elif key == 'C':
            self.backward_firing['c'] += sax_M

        elif key == 'O':
            self.backward_firing['o'] += sax_M

    def update_forward_firing(self, sax_i, sax_c, sax_o):
        self.forward_firing['i'] += sax_i.sum(axis=0)
        self.forward_firing['c'] += sax_c.sum(axis=0)
        self.forward_firing['o'] += sax_o.sum(axis=0)

    def update_mask(self, t):

        self.matrices['Im'] = self.clip_firing(self.matrices['Im'].multiply(self.I), self.backward_firing['i'], t)
        self.matrices['Cm'] = self.clip_firing(self.matrices['Cm'].multiply(self.C), self.backward_firing['c'], t)
        self.matrices['Om'] = self.clip_firing(self.matrices['Om'].multiply(self.O), self.backward_firing['o'], t)

    @staticmethod
    def clip_firing(sax_mask, sax_bf, t):
        for i, j in zip(*sax_mask.multiply(sax_bf).nonzero()):
            if sax_bf[i, j] >= t:
                sax_mask[i, j] = False
        return sax_mask

    @staticmethod
    def load_pickle(path, project, graph_id=None):

        with open(path, 'rb') as handle:
            d_graph = pickle.load(handle)

        return FiringGraph.from_dict(d_graph, project, graph_id=graph_id)

    @staticmethod
    def from_matrices(sax_I, sax_C, sax_O, ax_levels, mask_matrices=None, mask_vertices=None, depth=2, project='fg',
                      graph_id=None, partitions=None):
        """

        :param sax_I:
        :param sax_C:
        :param sax_O:
        :param ax_levels:
        :param mask_matrices:
        :param mask_vertices:
        :param depth:
        :param project:
        :param graph_id:
        :param partitions:
        :return:
        """

        if not any([mask_matrices is not None, mask_vertices is not None]):
            raise ValueError("Either mask should be set for vertices or direct link matrices")

        if mask_vertices is not None:
            mask_matrices = utils.mat_mask_from_vertice_mask(sax_I, sax_C, sax_O, mask_vertices)

        d_matrices = dict(list(mask_matrices.items()) + [('Iw', sax_I), ('Cw', sax_C), ('Ow', sax_O)])

        return FiringGraph(
            project, ax_levels, depth=depth, matrices=d_matrices, graph_id=graph_id, partitions=partitions
        )

    @staticmethod
    def from_dict(d_graph, project, graph_id=None):
        """

        :param d_graph:
        :param project:
        :param graph_id:
        :return:
        """
        fg = FiringGraph(
            project, d_graph['levels'], matrices=d_graph['matrices'], graph_id=graph_id,
            is_drained=d_graph['is_drained'], depth=d_graph['depth']
        )

        return fg

    def propagate(self, sax_i):
        """

        :param sax_i:
        :return:
        """

        # Init core signal to all zeros
        sax_c = csc_matrix((sax_i.shape[0], self.C.shape[0]))

        for i in range(self.depth - 1):

            # Core transmit
            sax_c = ftc(self.C, self.I, sax_c, sax_i)
            sax_c = fpc(sax_c, None, self.levels)

            if i == 0:
                sax_i = csc_matrix(sax_i.shape)

        sax_o = fto(self.O, sax_c)

        return sax_o

    def save_as_pickle(self, path):
        d_graph = self.to_dict()

        with open(path, 'wb') as handle:
            pickle.dump(d_graph, handle)

    def to_dict(self, is_copy=False):

        d_graph = {
            'is_drained': self.is_drained, 'is_dried': False, 'matrices': self.matrices, 'levels': self.levels,
            'depth': self.depth
        }

        if is_copy:
            d_graph.update({'matrices': copy.deepcopy(self.matrices), 'levels': self.levels.copy()})

        return d_graph

    def copy(self):
        return self.from_dict(self.to_dict(is_copy=True), self.project, self.graph_id)


def merge_firing_graph(l_firing_graph, n_inputs, n_outputs, key_partition=lambda x: x):
    """

    :param l_firing_graph:
    :param n_inputs:
    :param n_outputs:
    :param key_partition:
    :return:
    """

    try:
        assert(len(set([fg.depth for fg in l_firing_graph])) == 1)
        depth = l_firing_graph[0].depth

    except AssertionError:
        raise ValueError("Firing graph merge is possible only if all firing graph has the same depth.")

    l_partitions, n_core_current = [], 0
    sax_I, sax_C, sax_O, l_levels = lil_matrix((n_inputs, 0)), lil_matrix((0, 0)), lil_matrix((0, n_outputs)), []
    d_masks = {
        'Im': lil_matrix((n_inputs, 0), dtype=bool),
        'Cm': lil_matrix((0, 0), dtype=bool),
        'Om': lil_matrix((0, n_outputs), dtype=bool)
    }

    for firing_graph in l_firing_graph:

        # Set partitions
        l_partitions.append({
            'indices': [n_core_current + key_partition(i) for i in range(firing_graph.Cw.shape[1])],
            'depth': firing_graph.depth
        })

        if firing_graph.partitions is not None:
            l_partitions[-1].update({'partitions': firing_graph.partitions})

        n_core_current += firing_graph.Cw.shape[1]

        # Merge io matrices
        sax_I = hstack([sax_I, firing_graph.Iw.tolil()])
        sax_O = vstack([sax_O, firing_graph.Ow.tolil()])

        # Merge Core matrices
        sax_C_new = hstack([lil_matrix((firing_graph.Cw.shape[0], sax_C.shape[1])), firing_graph.Cw.tolil()])
        sax_C = hstack([sax_C, lil_matrix((sax_C.shape[0], firing_graph.Cw.shape[1]))])
        sax_C = vstack([sax_C, sax_C_new])

        # Merge io masks
        d_masks['Im'] = hstack([d_masks['Im'], firing_graph.Im.tolil()])
        d_masks['Om'] = vstack([d_masks['Om'], firing_graph.Om.tolil()])

        # Merge Core masks
        mask_C_new = hstack([lil_matrix((firing_graph.Cm.shape[0], d_masks['Cm'].shape[1])), firing_graph.Cm.tolil()])
        mask_C = hstack([d_masks['Cm'], lil_matrix((d_masks['Cm'].shape[0], firing_graph.Cm.shape[1]))])
        d_masks['Cm'] = vstack([mask_C, mask_C_new])

        # Merge levels
        l_levels.extend(list(firing_graph.levels))

    return FiringGraph.from_matrices(
        sax_I.tocsc(), sax_C.tocsc(), sax_O.tocsc(), array(l_levels), mask_matrices=d_masks, depth=depth,
        partitions=l_partitions
    )


def extract_structure(partition, firing_graph):
    """

    :param partition:
    :param firing_graph:
    :return:
    """
    l_ind_partition = partition['mapping'].values()

    sax_I = firing_graph.Iw[:, l_ind_partition]
    sax_C = firing_graph.Cw[l_ind_partition, :][:, l_ind_partition]
    sax_O = firing_graph.Ow[l_ind_partition, :]

    d_masks = {
        'Im': firing_graph.Im[:, l_ind_partition],
        'Cm': firing_graph.Cm[l_ind_partition, :][:, l_ind_partition],
        'Om': firing_graph.Om[l_ind_partition, :]
    }

    ax_levels = firing_graph.levels[l_ind_partition]

    return FiringGraph.from_matrices(
        sax_I.tocsc(), sax_C.tocsc(), sax_O.tocsc(), ax_levels, mask_matrices=d_masks, depth=partition['depth'],
        partitions=partition.get('partitions', None)
    )
