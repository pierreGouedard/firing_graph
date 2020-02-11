# Global imports
import pickle
import random
import string
from scipy.sparse import lil_matrix, csc_matrix, hstack, vstack
import copy
from numpy import uint32

# Local import
from ..data_structure import utils
from ..tools.equations.forward import ftc, fto, fpc


class FiringGraph(object):
    """
    This class implement the main data structure used for fiting data. It is composed of weighted link in the form of
    scipy.sparse matrices and store complement information on vertices such as levels, mask for draining. It also keep
    track of the firing of vertices.

    """
    def __init__(self, project, ax_levels, matrices, depth=2, graph_id=None, is_drained=False, partitions=None,
                 drainer_params=None, precision=None, backward_firing=None):

        if graph_id is None:
            graph_id = ''.join([random.choice(string.ascii_letters) for _ in range(5)])

        # Utils
        self.project = project
        self.graph_id = graph_id
        self.is_drained = is_drained

        # architecture core params
        self.depth = depth
        self.levels = ax_levels

        # force format and type of matrices
        utils.set_matrices_spec(matrices, write_mode=False)
        self.matrices = matrices

        # set additional (optional) util attribute
        self.partitions = partitions
        self.drainer_params = drainer_params if drainer_params is not None else {}
        self.precision = precision

        # Set backward tracking matrices
        if backward_firing is None:
            self.backward_firing = utils.create_empty_backward_firing(
                matrices['Iw'].shape[0], matrices['Ow'].shape[1], matrices['Cw'].shape[0]
            )
        else:
            self.backward_firing = {k: sax_bf.astype(uint32).tocsc() for k, sax_bf in backward_firing}

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
            self.backward_firing['i'] += sax_M.astype(self.backward_firing['i'].dtype)

        elif key == 'C':
            self.backward_firing['c'] += sax_M.astype(self.backward_firing['c'].dtype)

        elif key == 'O':
            self.backward_firing['o'] += sax_M.astype(self.backward_firing['o'].dtype)

    def update_mask(self, t):
        if self.matrices['Im'].nnz > 0:
            self.matrices['Im'] = self.matrices['Im'].multiply(self.I)
            if t > 0:
                self.matrices['Im'] = self.clip_firing(self.matrices['Im'].tolil(), self.backward_firing['i'], t)

        if self.matrices['Cm'].nnz > 0:
            self.matrices['Cm'] = self.matrices['Cm'].multiply(self.C)
            if t > 0:
                self.matrices['Cm'] = self.clip_firing(self.matrices['Cm'], self.backward_firing['c'], t)

        if self.matrices['Om'].nnz > 0:
            self.matrices['Om'] = self.matrices['Om'].multiply(self.O)
            if t > 0:
                self.matrices['Om'] = self.clip_firing(self.matrices['Om'], self.backward_firing['o'], t)

    @staticmethod
    def clip_firing(sax_mask, sax_bf, t):
        for i, j in zip(*sax_mask.nonzero()):
            if sax_bf[i, j] >= t:
                sax_mask[i, j] = False
        return sax_mask.tocsc()

    @staticmethod
    def load_pickle(path):

        with open(path, 'rb') as handle:
            d_graph = pickle.load(handle)

        return FiringGraph.from_dict(d_graph)

    @staticmethod
    def from_matrices(sax_I, sax_C, sax_O, ax_levels, mask_matrices=None, mask_vertices=None, depth=2, project='fg',
                      graph_id=None, partitions=None, drainer_params=None, precision=None):
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
        :param drainer_params:
        :return:
        """

        if not any([mask_matrices is not None, mask_vertices is not None]):
            raise ValueError("Either mask should be set for vertices or direct link matrices")

        if mask_vertices is not None:
            mask_matrices = utils.mat_mask_from_vertice_mask(sax_I, sax_C, sax_O, mask_vertices)

        d_matrices = dict(list(mask_matrices.items()) + [('Iw', sax_I), ('Cw', sax_C), ('Ow', sax_O)])

        return FiringGraph(
            project, ax_levels, depth=depth, matrices=d_matrices, graph_id=graph_id, partitions=partitions,
            drainer_params=drainer_params, precision=precision
        )

    @staticmethod
    def from_dict(d_graph):
        """

        :param d_graph:
        :return:
        """
        return FiringGraph(**d_graph)

    def propagate(self, sax_i):
        """

        :param sax_i:
        :return:
        """

        # Init core signal to all zeros
        sax_c = csc_matrix((sax_i.shape[0], self.C.shape[0]))

        for i in range(self.depth - 1):

            # Core transmit
            sax_c = ftc(self.C, self.I, sax_c, sax_i.astype(int))
            sax_c = fpc(sax_c, None, self.levels)

            if i == 0:
                sax_i = csc_matrix(sax_i.shape)

        sax_o = fto(self.O, sax_c)

        return sax_o > 0

    def save_as_pickle(self, path):
        d_graph = self.to_dict()

        with open(path, 'wb') as handle:
            pickle.dump(d_graph, handle)

    def to_dict(self, deep_copy=False):

        d_graph = {
            'project': self.project,
            'graph_id': self.graph_id,
            'is_drained': self.is_drained,
            'matrices': self.matrices,
            'ax_levels': self.levels,
            'depth': self.depth,
            'partitions': self.partitions,
            'precision': self.precision,
        }

        if deep_copy:
            d_graph.update({'matrices': copy.deepcopy(self.matrices), 'ax_levels': self.levels.copy()})

        return d_graph

    def copy(self):
        return self.from_dict(self.to_dict(deep_copy=True))
