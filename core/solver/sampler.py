# Global import
import numpy as np
from scipy.sparse import csc_matrix

# local import
from ..data_structure.graph import FiringGraph


class SupervisedSampler(object):
    """

    """
    def __init__(self, imputer, n_inputs, n_outputs, p_sample=0.8, n_vertices=10, l0=1, firing_graph=None,
                 structures=None, max_iter=1000, verbose=0, output_negation=False):
        """
        :param imputer: imputer (see core.tools.imputers)
        :param n_inputs: list [#input, #output]
        :param n_outputs: list [#input, #output]
        :param p_sample: float probability of sampling
        :param n_vertices: int
        :param firing_graph:
        :param structures:
        :param verbose: int
        :param output_negation: bool
        """
        # size of problem
        self.n_inputs, self.n_outputs = n_inputs, n_outputs

        # Sampling parameters
        self.p_sample, self.n_vertices, self.output_negation = p_sample, n_vertices, output_negation

        # Utils parameters
        self.verbose, self.max_iter = verbose, max_iter

        # Structure  parameter
        self.l0 = l0

        # Core attributes
        self.firing_graph = firing_graph
        self.structures = structures
        self.vertices = None
        self.imputer = imputer

    def generative_sampling(self):

        # Initialisation
        self.vertices = {i: [] for i in range(self.n_outputs)}
        ax_selected, n = np.zeros(self.n_outputs, dtype=int), 0

        # Core loop
        while ax_selected.sum() != len(ax_selected) * self.n_vertices:

            # Get forward and backward signals
            sax_i = self.imputer.next_forward()
            sax_got = self.imputer.next_backward()

            # Propagate through firing graph
            if self.firing_graph is not None:
                sax_o = self.firing_graph.propagate(sax_i)
            else:
                sax_o = csc_matrix((1, self.n_outputs))

            for i in sax_got.nonzero()[1]:
                if ax_selected[i] < self.n_vertices and sax_o[0, i] == 0:

                    # Randomly sample active bits
                    ax_indices = np.array(sax_i.nonzero()[1])
                    ax_mask = np.random.binomial(1, self.p_sample, len(ax_indices))

                    # Add sampled bits to list of output i's vertices
                    if ax_mask.any():
                        self.vertices[i].append([set(ax_indices[ax_mask > 0])])
                        ax_selected[i] += 1

            n += 1

            if n > self.max_iter:
                break

        return self

    def discriminative_sampling(self):

        # Init
        self.vertices = {i: [] for i in range(len(self.structures))}
        ax_selected, n = np.zeros(len(self.structures), dtype=int), 0

        # Core loop
        while ax_selected.sum() != len(ax_selected) * self.n_vertices:

            # Get forward and backward signals
            sax_i = self.imputer.next_forward()
            sax_got = self.imputer.next_backward()

            # Propagate through firing graph
            if self.firing_graph is not None:
                sax_o = self.firing_graph.propagate(sax_i)
            else:
                sax_o = csc_matrix((1, self.n_outputs))

            # For each output and each structure, sample active bits
            for i in sax_got.nonzero()[1]:
                for j, structure in enumerate(self.structures):
                    if ax_selected[j] < self.n_vertices and structure.propagate(sax_i)[0, i] > 0 and sax_o[0, i] == 0:

                        # Randomly samples active bits
                        ax_indices = np.array(sax_i.nonzero()[1])
                        ax_mask = np.random.binomial(1, self.p_sample, len(ax_indices))

                        # Add sampled bits to list of output i vertices
                        if ax_mask.any():
                            self.vertices[j] += [set(ax_indices[ax_mask > 0])]
                            ax_selected[j] += 1

            n += 1

            if n > self.max_iter:
                break

        return self

    def build_structures(self, weight):
        """

        :return:
        """

        if self.structures is None:
            for i in range(self.n_outputs):
                self.structures.extend(self.create_structures(i, weight))
        else:
            l_structures = []
            for i, structure in enumerate(self.structures):
                l_structures.extend(self.augment_structures(i, structure, weight))

            self.structures = l_structures

        #firing_graph = merge_structures(self.strucures)

        return self

    def create_structures(self, i, w):
        """

        :param i:
        :param w:
        :return:
        """
        # Init
        n_core = self.n_vertices + 1
        sax_I, sax_C, ax_levels = csc_matrix((self.n_inputs, 1)), csc_matrix((n_core, n_core)), np.array([self.l0])
        d_mask = {'I': np.zeros(self.n_inputs), 'C': np.zeros(n_core), 'O': np.zeros(self.n_outputs)}

        # Core loop
        for j, l_bits in enumerate(self.vertices[i]):
            for bit in l_bits:
                sax_I[bit, j] = w
                d_mask['I'][bit] = 1
                sax_C[j, n_core - 1] = 1

            # Set Output connection and level
            sax_O = csc_matrix((n_core, self.n_outputs))
            sax_O[n_core - 1, i] = 1

            yield FiringGraph.from_matrices(sax_I, sax_C, sax_O, ax_levels, d_mask, depth=2)

    def augment_structures(self, i, structure, w):
        """

        :param i:
        :param structure:
        :param w:
        :return:
        """
        # Init
        n_core = structure.Cw.shape[0] + self.n_vertices + 2
        sax_I, sax_C = structure.Iw, csc_matrix((n_core, n_core))
        d_mask = {'I': np.zeros(self.n_inputs), 'C': np.zeros(n_core), 'O': np.zeros(self.n_outputs)}

        # Set level and init link matrix
        # TODO since update of links is set at the bit level make sure non fitted edges has large values so to ensure it will not break with draining
        sax_I[:, :structure.Iw.shape[1]] = structure.Iw * 10000000
        sax_C[structure.Cw.shape[0]:, structure.Cw.shape[0]] = structure.Cw
        ax_levels = np.array(list(structure.levels) + [self.l0] * self.n_vertices + [1, 2])

        # Core loop
        for j, l_bits in enumerate(self.vertices[i]):
            for bit in l_bits:
                sax_I[bit, structure.Cw.shape[0] + j] = w
                sax_C[structure.Cw.shape[0] + j, n_core - 2] = 1
                structure.mask['I'][bit] = 1

        # Add core edges
        sax_C[structure.Cw.shape[0] - 1, n_core - 1] = 1
        sax_C[n_core - 2, n_core - 1] = 1

        # Set outputs
        sax_O = csc_matrix((n_core, self.n_outputs))
        sax_O[n_core - 1, structure.Ow.nonzeros()[1]] = 1

        return FiringGraph.from_matrices(sax_I, sax_C, sax_O, ax_levels, d_mask, depth=4)
