# Global imports
import pickle
import random
import string
from scipy.sparse import lil_matrix
import copy

# Local import
from ..data_structure import utils
from ..tools.drivers.driver import FileDriver

driver = FileDriver('graph_file_driver', '')


class FiringGraph(object):
    """
    This class implement the main data structure used for fiting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """
    def __init__(self, project, ax_levels, matrices, depth=2, graph_id=None, is_drained=False):

        if graph_id is None:
            graph_id = ''.join([random.choice(string.ascii_letters) for _ in range(5)])

        self.project = project
        self.graph_id = graph_id
        self.is_drained = is_drained

        # Parameter
        self.depth = depth
        self.levels = ax_levels
        self.matrices = matrices

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
    def from_edges(project, ni, nc, no, l_edges, weights, ax_levels, mask_vertice_drain, depth=2, graph_id=None):
        """

        :param project:
        :param ni:
        :param nc:
        :param no:
        :param l_edges:
        :param weights:
        :param ax_levels:
        :param mask_vertice_drain:
        :param depth:
        :param graph_id:
        :return:
        """

        # Get adjacency matrices
        sax_I, sax_C, sax_O = utils.mat_from_tuples(ni, no, nc, l_edges, weights)

        # Get dainer mask
        d_mask = utils.mat_mask_from_vertice_mask(sax_I, sax_C, sax_O, mask_vertice_drain)

        # Init dict of structure of firing graph
        d_matrices = {'Iw': sax_I, 'Cw': sax_C, 'Ow': sax_O}
        d_matrices.update(d_mask)

        return FiringGraph(project, ax_levels, depth=depth, matrices=d_matrices, graph_id=graph_id)

    @staticmethod
    def from_matrices(project, sax_I, sax_C, sax_O, ax_levels, mask_vertice_drain, depth=2, graph_id=None):
        """

        :param project:
        :param sax_I:
        :param sax_C:
        :param sax_O:
        :param ax_levels:
        :param mask_vertice_drain:
        :param depth:
        :param graph_id:
        :return:
        """

        # Get dainer mask matrices
        d_mask = utils.mat_mask_from_vertice_mask(sax_I, sax_C, sax_O, mask_vertice_drain)
        d_matrices = dict(list(d_mask.items()) + [('Iw', sax_I), ('Cw', sax_C), ('Ow', sax_O)])

        return FiringGraph(project, ax_levels, depth=depth, matrices=d_matrices, graph_id=graph_id)

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
