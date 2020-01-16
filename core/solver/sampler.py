# Global import
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix

# local import
from ..data_structure.graph import FiringGraph, merge_firing_graph


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
        """
        Sampling to create first Structure.

        :return:
        """
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
                    ax_indices = np.array(sax_i.nonzero()[0])
                    ax_mask = np.random.binomial(1, self.p_sample, len(ax_indices))

                    # Add sampled bits to list of output i's vertices
                    if ax_mask.any():
                        self.vertices[i].extend([set(ax_indices[ax_mask > 0])])
                        ax_selected[i] += 1

            n += 1

            if n > self.max_iter:
                break

        return self

    def discriminative_sampling(self):
        """
        Sampling to complete Structure.

        :return:
        """
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
                    l_struct_indices = list(structure.I.nonzero()[0])

                    if ax_selected[j] < self.n_vertices and structure.propagate(sax_i)[0, i] > 0 and sax_o[0, i] == 0:
                        ax_indices = np.array([ind for ind in sax_i.nonzero()[0] if ind not in l_struct_indices])
                        ax_mask = np.random.binomial(1, self.p_sample, len(ax_indices))

                        # Add sampled bits to list of output i vertices
                        if ax_mask.any():
                            self.vertices[j].extend([set(ax_indices[ax_mask > 0])])
                            ax_selected[j] += 1

            n += 1

            if n > self.max_iter:
                break

        return self

    def build_firing_graph(self, l_weights):
        """

        :return:
        """

        if self.vertices is None:
            raise ValueError(
                "Before Building firing graph, one need to sample input bits using generative or discriminative "
                "sampling"
            )

        if self.structures is None:
            self.structures = []
            for i in range(self.n_outputs):
                self.structures.append(self.create_structures(i, l_weights[i]))
        else:
            l_structures = []
            for i, structure in enumerate(self.structures):
                l_structures.append(self.augment_structures(i, structure, l_weights[i]))

            self.structures = l_structures

        firing_graph = merge_firing_graph(self.structures, self.n_inputs, self.n_outputs)

        return firing_graph

    def create_structures(self, i, w):
        """

        :param i:
        :param w:
        :return:
        """
        # Init
        n_core = self.n_vertices + 1
        sax_I, sax_C, ax_levels = lil_matrix((self.n_inputs, n_core)), lil_matrix((n_core, n_core)), np.array([self.l0])
        d_mask = {
            'Im': lil_matrix((self.n_inputs, n_core), dtype=bool),
            'Cm': lil_matrix((n_core, n_core), dtype=bool),
            'Om': lil_matrix((n_core, self.n_outputs), dtype=bool)
        }

        # Core loop
        for j, l_bits in enumerate(self.vertices[i]):
            for bit in l_bits:
                sax_I[bit, j] = w
                d_mask['Im'][bit, j] = True
                sax_C[j, n_core - 1] = 1

        # Set Output connection and level
        sax_O = lil_matrix((n_core, self.n_outputs))
        sax_O[n_core - 1, i] = 1

        return FiringGraph.from_matrices(
            sax_I.tocsc(), sax_C.tocsc(), sax_O.tocsc(), ax_levels, mask_matrices=d_mask, depth=3
        )

    def augment_structures(self, i, structure, w):
        """

        :param i:
        :param structure:
        :param w:
        :param w_max:
        :return:
        """
        # Init
        n_core = structure.Cw.shape[0] + self.n_vertices + 3
        sax_I, sax_C = lil_matrix((self.n_inputs, n_core)), lil_matrix((n_core, n_core))
        d_mask = {
            'Im': lil_matrix((self.n_inputs, n_core), dtype=bool),
            'Cm': lil_matrix((n_core, n_core), dtype=bool),
            'Om': lil_matrix((n_core, self.n_outputs), dtype=bool)
        }

        # Set level and init link matrix
        sax_I[:, :structure.Iw.shape[1]] = structure.Iw.tolil()
        sax_C[:structure.Cw.shape[0], :structure.Cw.shape[0]] = structure.Cw.tolil()
        ax_levels = np.array(list(structure.levels) + [self.l0] * self.n_vertices + [1, 1, 2])

        # Core loop
        for j, l_bits in enumerate(self.vertices[i]):
            for bit in l_bits:
                sax_I[bit, structure.Cw.shape[0] + j] = w
                sax_C[structure.Cw.shape[0] + j, n_core - 3] = 1
                sax_C[n_core - 3, n_core - 2] = 1
                d_mask['Im'][bit, structure.Cw.shape[0] + j] = True

        # Add core edges
        sax_C[structure.Cw.shape[0] - 1, n_core - 1] = 1
        sax_C[n_core - 2, n_core - 1] = 1

        # Set outputs
        sax_O = lil_matrix((n_core, self.n_outputs))
        sax_O[n_core - 1, int(structure.Ow.nonzero()[1])] = 1

        # Set partitions
        partitions = [
            {
                'indices': range(structure.Cw.shape[1]),
                'name': "base",
                "depth": 4
            },
            {
                'indices': [structure.Cw.shape[1] + i for i in range(sax_C.shape[1] - structure.Cw.shape[1] - 2)],
                "name": "transient",
                "depth": 3
            },
        ]

        return FiringGraph.from_matrices(
            sax_I.tocsc(), sax_C.tocsc(), sax_O.tocsc(), ax_levels, mask_matrices=d_mask, depth=5, partitions=partitions
        )
