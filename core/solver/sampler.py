# Global import
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix, hstack, vstack

# local import
from ..data_structure.graph import FiringGraph
from ..data_structure.structures import StructureIntersection
from ..data_structure.utils import create_empty_matrices, augment_matrices


class SupervisedSampler(object):
    """

    """
    def __init__(self, imputer, n_inputs, n_outputs, batch_size, p_sample=0.8, n_vertices=10, l0=1, firing_graph=None,
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
        self.n_inputs, self.n_outputs, self.batch_size = n_inputs, n_outputs, batch_size

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

    def get_signal_batch(self):
        sax_i, sax_o = csr_matrix((0, self.n_inputs), dtype=bool), csr_matrix((0, self.n_outputs), dtype=bool)

        for _ in range(self.batch_size):
            sax_i = vstack([sax_i, self.imputer.next_forward().tocsr()])
            sax_o = vstack([sax_o, self.imputer.next_backward().tocsr()])

        return sax_i, sax_o

    def generative_sampling(self):

        # Init
        self.vertices = {i: [] for i in range(self.n_outputs)}
        ax_selected, n = np.zeros(self.n_outputs, dtype=int), 0

        sax_i, sax_got = self.get_signal_batch()

        for i in range(self.n_outputs):
            if self.firing_graph is not None:
                sax_fg = self.firing_graph.propagate(sax_i)[:, i]
            else:
                sax_fg = csc_matrix((self.batch_size, 1), dtype=bool)

            # selected random active
            ax_mask = sax_got[:, i].toarray()[:, 0] & ~(sax_fg.toarray()[:, 0])
            n_sample = min(ax_mask.sum(), self.n_vertices)

            if n_sample == 0:
                continue

            l_sampled_indices = np.random.randint(n_sample, size=n_sample)
            sax_samples = sax_i[ax_mask, :][l_sampled_indices, :]

            for sax_sample in sax_samples:
                ax_indices = sax_sample.nonzero()[1]
                ax_mask = np.random.binomial(1, self.p_sample, len(ax_indices))

                # Add sampled bits to list of output i vertices
                if ax_mask.any():
                    self.vertices[i].extend([set(ax_indices[ax_mask > 0])])
                    ax_selected[i] += 1

        print("[Sampler]: Generative sampling has sampled {} vertices".format(ax_selected.sum()))

        return self

    def discriminative_sampling(self):
        # Init
        self.vertices = {i: [] for i in range(len(self.structures))}
        ax_selected, n = np.zeros(len(self.structures), dtype=int), 0

        sax_i, sax_got = self.get_signal_batch()

        for i in range(self.n_outputs):
            if self.firing_graph is not None:
                sax_fg = self.firing_graph.propagate(sax_i)[:, i]
            else:
                sax_fg = csc_matrix((self.batch_size, 1), dtype=bool)

            l_structures_sub = [(j, struct) for j, struct in enumerate(self.structures) if i in struct.O.nonzero()[1]]

            for j, struct in l_structures_sub:

                l_struct_indices = list(set(struct.I.nonzero()[0]))
                sax_struct = struct.propagate(sax_i)[:, i]

                # selected random active
                ax_mask = sax_struct.multiply(sax_got[:, i]).toarray()[:, 0] & ~(sax_fg.toarray()[:, 0])
                n_sample = min(ax_mask.sum(), self.n_vertices)

                if n_sample == 0:
                    continue

                l_sampled_indices = np.random.randint(n_sample, size=n_sample)
                sax_samples = sax_i[ax_mask, :][l_sampled_indices, :]

                for sax_sample in sax_samples:
                    ax_indices = np.array([ind for ind in sax_sample.nonzero()[1] if ind not in l_struct_indices])
                    ax_mask = np.random.binomial(1, self.p_sample, len(ax_indices))

                    # Add sampled bits to list of output i vertices
                    if ax_mask.any():
                        self.vertices[j].extend([set(ax_indices[ax_mask > 0])])
                        ax_selected[j] += 1

        print("[Sampler]: Discriminative sampling has sampled {} vertices".format(ax_selected.sum()))

        return self

    def build_firing_graph(self, drainer_params, return_structures=False):
        """

        :return:
        """

        if self.vertices is None:
            raise ValueError(
                "Before Building firing graph, one need to sample input bits using generative or discriminative "
                "sampling"
            )

        l_structures = []
        if self.structures is None:
            for i in range(self.n_outputs):

                # Set partition
                partitions = [
                    {'indices': [], 'name': "base", "depth": 4},
                    {'indices': range(self.n_vertices), "name": "transient", "depth": 3},
                ]

                # Add structure
                l_structures.append(StructureIntersection.from_input_indices(
                    self.n_inputs, self.n_outputs, np.array([self.l0] * (self.n_vertices + 1)), i,
                    self.vertices[i], drainer_params['weight'], **{"partitions": partitions}
                ))

        else:
            for i, structure in enumerate(self.structures):

                # Set partitions
                partitions = [
                    {'indices': range(structure.n_intersection), 'name': "base", 'precision': structure.precision,
                     "depth": 2},
                    {'indices': [structure.n_intersection + j for j in range(self.n_vertices)], "name": "transient",
                     "depth": 3},
                ]

                # Add structure
                l_structures.append(structure.augment_structure(
                    self.n_vertices, self.vertices[i], np.array([self.l0] * self.n_vertices), drainer_params['weight'],
                    **{"partitions": partitions}
                ))

        firing_graph = self.merge_structures(l_structures, drainer_params)

        if return_structures:
            return firing_graph, l_structures

        return firing_graph

    def merge_structures(self, l_structures, drainer_params):
        """
    
        :param l_structures:
        :return:
        """

        if len(l_structures) == 0:
            return None

        # Make sure all graph has the same depth
        assert len(set([fg.depth for fg in l_structures])) == 1, "Firing graph merge is possible only if all firing " \
                                                                 "graph has the same depth."

        l_partitions, n_core_current, l_levels, depth = [], 0, [], l_structures[0].depth
        d_matrices = create_empty_matrices(self.n_inputs, self.n_outputs, 0)

        for structure in l_structures:
    
            # Set partitions
            l_partitions.append({
                'indices': [n_core_current + i for i in range(structure.Cw.shape[1])],
                'depth': structure.depth,
                'index_output': structure.Ow.nonzero()[1][0]
            })
    
            if structure.partitions is not None:
                l_partitions[-1].update({'partitions': structure.partitions})

            if structure.precision is not None:
                l_partitions[-1].update({'precision': structure.precision})

            n_core_current += structure.Cw.shape[1]

            d_matrices = augment_matrices(d_matrices, structure.matrices)

            # Merge levels
            l_levels.extend(list(structure.levels))

        return FiringGraph(
            'fg', np.array(l_levels), d_matrices, depth=depth, partitions=l_partitions, drainer_params=drainer_params
        )

    def merge_firing_graph(self, firing_graph):
        if self.firing_graph is None:
            self.firing_graph = firing_graph
            return

        # Make sure firing graph has the same depth as existing one
        assert firing_graph.depth == self.firing_graph.depth, 'Non compatible depth'

        # merge partitions
        if firing_graph.partitions is not None:
            l_partitions = [
                {
                    'indices': [self.firing_graph.Cw.shape[0] + i for i in partition['indices']],
                    'depth': partition['depth'],
                    'precision': partition['precision'],
                    'index_output': partition['index_output']
                }
                for partition in firing_graph.partitions
            ]

            if self.firing_graph.partitions is not None:
                self.firing_graph.partitions.extend(l_partitions)

            else:
                self.firing_graph.partitions = l_partitions

        # merge matrices
        self.firing_graph.matrices = augment_matrices(self.firing_graph.matrices, firing_graph.matrices)

        # Merge levels
        self.firing_graph.levels = np.hstack((self.firing_graph.levels, firing_graph.levels))

        return self
