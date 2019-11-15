# Global import
import numpy as np

# local import
from core.data_structure.utils import mat_from_tuples
from core.data_structure.graph import FiringGraph


class Sampler(object):
    # Firing Graph of depth 2
    depth_init = 2

    # Firing Graph of depth 3
    depth_core = 3

    def __init__(self, size, w, imputer, p_sample=1, selected_bits=None, preselected_bits=None, cores=None, supervised=True, verbose=0):
        """

        :param size: list [#input, #output]
        :param w: int weights of edges of firing graph
        :param p_sample: float probability of sampling
        :param imputer: deyep.core.imputer.comon.Imputer
        :param selected_bits: dict of set of inputs index already sampled in previous iteration (key = output index)
        :param preselected_bits: dict of set of inputs index from which we want to draw next sample (key = output index)
        :param supervised: bool
        :param verbose: int
        """
        # Core params
        self.ni, self.no = size[0], size[1]
        self.w = w
        self.p_sample = p_sample
        self.firing_graph = None
        self.verbose = verbose
        self.supervised = supervised
        self.core_vertices = cores if cores is not None else {}

        # Get list of preselected and already selected bits if any
        if preselected_bits is None:
            self.preselect_bits = {}
        else:
            self.preselect_bits = preselected_bits

        if selected_bits is None:
            self.selected_bits = {}
        else:
            self.selected_bits = selected_bits

        # utils
        self.imputer = imputer

    def reset_imputer(self):
        self.imputer.stream_features()
        return self

    def reset_firing_graph(self):
        self.core_vertices = {}
        self.firing_graph = None

    def sample(self):

        if len(self.preselect_bits) == 0:
            if self.supervised:
                self.sample_supervised()
            else:
                raise NotImplementedError

        return self

    def sample_supervised(self):

        ax_selected = np.zeros(self.no, dtype=bool)

        # Select bits for each output
        while ax_selected.sum() != len(ax_selected):
            sax_si = self.imputer.stream_next_forward()
            sax_got = self.imputer.stream_next_backward()

            for i in sax_got.nonzero()[1]:
                if not ax_selected[i]:
                    if self.p_sample < 1.:
                        ax_indices = np.array(sax_si.nonzero()[1])
                        ax_mask = np.random.binomial(1, self.p_sample, len(ax_indices))
                        self.preselect_bits[i] = set(ax_indices[ax_mask > 0])

                    else:
                        self.preselect_bits[i] = set(sax_si.nonzero()[1])

                    # Remove already selected bits
                    if len(self.selected_bits) > 0:
                        self.preselect_bits[i] = self.preselect_bits[i]\
                            .difference(set(self.selected_bits.get(i, {})))

                    ax_selected[i] = True

        return self

    def build_graph_multiple_output(self, name='sampler'):
        """

        :return:
        """
        # Init parameter of the firing graph
        l_edges, d_mask, n_core, d_levels = [], {'I': np.zeros(self.ni)}, 0, {}

        if len(self.selected_bits) == 0:
            depth = Sampler.depth_init
        else:
            depth = Sampler.depth_core

        # Build matrices of the firing graph
        for i in range(self.no):
            l_edges, d_mask, n_core, d_levels = self.build_graph(i, l_edges, d_mask, n_core, d_levels)

        sax_I, sax_C, sax_O = mat_from_tuples(self.ni, self.no, n_core, l_edges, self.w)

        # Build level array
        ax_levels = np.zeros(n_core)
        for i, v in d_levels.items():
            ax_levels[i] = v

        # Complete mask
        d_mask['C'] = np.zeros(n_core)

        # Build firing graph
        self.firing_graph = FiringGraph.from_matrices(name, sax_I, sax_C, sax_O, ax_levels, d_mask, depth)

        return self

    def build_graph(self, i, l_edges, d_mask, n_core, d_levels):

        # Create first layer of graph (input vertices)
        for pb in self.preselect_bits[i]:
            l_edges += [('input_{}'.format(pb), 'core_{}'.format(n_core))]
            d_mask['I'][pb] = 1

        # If some bits have already been selected
        if self.selected_bits:
            for b in self.selected_bits[i]:
                l_edges += [('input_{}'.format(b), 'core_{}'.format(n_core + 1))]

            # Add core edges
            l_edges += [
                ('core_{}'.format(n_core), 'core_{}'.format(n_core + 2)),
                ('core_{}'.format(n_core + 1), 'core_{}'.format(n_core + 2)),
                ('core_{}'.format(n_core + 2), 'output_{}'.format(i))
            ]

            # Update levels
            d_levels.update({n_core: 1, n_core + 1: len(self.selected_bits[i]), n_core + 2: 2})
            self.core_vertices.update({i: ['core_{}'.format(n_core + j) for j in range(3)]})
            n_core += 3

        else:
            # Add core edges and update levels
            l_edges += [('core_{}'.format(n_core), 'output_{}'.format(i))]
            d_levels.update({n_core: 1})
            self.core_vertices.update({i: ['core_{}'.format(n_core)]})
            n_core += 1

        return l_edges, d_mask, n_core, d_levels
