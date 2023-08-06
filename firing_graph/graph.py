# Global imports
import pickle
import random
import string
from scipy.sparse import vstack, lil_matrix, csr_matrix
import copy
from numpy import int32

# Local import
from .linalg.forward import fti, ftc, fto


class FiringGraph(object):
    def __init__(
            self, project, ax_levels, matrices, depth=2, graph_id=None, meta=None,
    ):

        if graph_id is None:
            graph_id = ''.join([random.choice(string.ascii_letters) for _ in range(5)])

        # Utils
        self.project = project
        self.graph_id = graph_id

        # architecture src params
        self.depth = depth
        self.levels = ax_levels

        # Save core data & metadata
        self.matrices = matrices
        self.refresh_matrices()
        self.meta = meta

        # Set backward tracking matrices
        self.backward_firing = csr_matrix(matrices['I'].shape, dtype=int32)

    @property
    def C(self):
        return self.matrices['C']

    @property
    def O(self):
        return self.matrices['O']

    @property
    def I(self):
        return self.matrices['I']

    @property
    def Iw(self):
        return self.matrices['Iw']

    @property
    def Im(self):
        return self.matrices['Im']

    def reset_backward_firing(self):
        self.backward_firing = lil_matrix(self.backward_firing.shape, dtype=self.backward_firing.dtype)
        return self

    def update_backward_count(self, sax_M):
        self.backward_firing += sax_M

    def refresh_matrices(self):
        self.matrices['I'] = self.matrices['Iw'] > 0

    @staticmethod
    def load_pickle(path):
        with open(path, 'rb') as handle:
            d_graph = pickle.load(handle)

        return FiringGraph.from_dict(d_graph)

    @staticmethod
    def from_dict(d_graph):
        return FiringGraph(**d_graph)

    def seq_propagate(self, sax_i, max_bs=5000000, fto_required=False):

        # Split works if toot
        if sax_i.shape[0] > max_bs:
            l_parts = [(max_bs * i, max_bs * (i + 1)) for i in range(int(sax_i.shape[0] / max_bs))]
            l_outputs = [self.seq_propagate(sax_i[i:j, :]) for i, j in l_parts]

            if j < sax_i.shape[0] - 1:
                l_outputs.append(self.seq_propagate(sax_i[j:, :]))

            return vstack(l_outputs)

        # Propagate input
        sax_c = fti(self.I, sax_i, self.levels)

        # Core transmit
        for i in range(self.depth - 2):
            sax_c = ftc(self.C, sax_c, self.levels)

        if fto_required:
            sax_o = fto(self.O, sax_c)
            return sax_o
        else:
            return sax_c

    def save_as_pickle(self, path):
        d_graph = self.to_dict()

        with open(path, 'wb') as handle:
            pickle.dump(d_graph, handle)

    def to_dict(self, deep_copy=False):

        d_graph = {
            'project': self.project,
            'graph_id': self.graph_id,
            'matrices': self.matrices,
            'ax_levels': self.levels,
            'depth': self.depth,
            'meta': self.meta
        }

        if deep_copy:
            d_graph.update({
                'matrices': copy.deepcopy(self.matrices), 'ax_levels': self.levels.copy(),
                'meta': copy.deepcopy(self.meta)
            })

        return d_graph

    def copy(self):
        return self.from_dict(self.to_dict(deep_copy=True))


def create_empty_matrices(n_inputs, n_outputs, n_core, write_mode=True):
    if write_mode:
        d_matrices = {
            'I': lil_matrix((n_inputs, n_core), dtype=bool),
            'C': lil_matrix((n_core, n_core), dtype=bool),
            'O': lil_matrix((n_core, n_outputs), dtype=bool),
            'Iw': lil_matrix((n_inputs, n_core), dtype=int32),
            'Im': lil_matrix((n_inputs, n_core), dtype=bool),
        }

    else:
        d_matrices = {
            'I': csr_matrix((n_inputs, n_core), dtype=bool),
            'C': csr_matrix((n_core, n_core), dtype=bool),
            'O': csr_matrix((n_core, n_outputs), dtype=bool),
            'Iw': csr_matrix((n_inputs, n_core), dtype=int32),
            'Im': csr_matrix((n_inputs, n_core), dtype=bool),
        }

    return d_matrices
