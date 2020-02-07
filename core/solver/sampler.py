# Global import
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, vstack

# local import


class SupervisedSampler(object):
    """

    """
    def __init__(self, imputer, n_inputs, n_outputs, batch_size, p_sample=0.8, n_vertices=10, l0=1, firing_graph=None,
                 base_patterns=None, verbose=0, output_negation=False):
        """
        :param imputer: imputer (see core.tools.imputers)
        :param n_inputs: list [#input, #output]
        :param n_outputs: list [#input, #output]
        :param p_sample: float probability of sampling
        :param n_vertices: int
        :param firing_graph:
        :param base_patterns:
        :param verbose: int
        :param output_negation: bool
        """
        # size of problem
        self.n_inputs, self.n_outputs, self.batch_size = n_inputs, n_outputs, batch_size

        # Sampling parameters
        self.p_sample, self.n_vertices, self.output_negation = p_sample, n_vertices, output_negation

        # Utils parameters
        self.verbose = verbose

        # Structure  parameter
        self.l0 = l0

        # Core attributes
        self.firing_graph = firing_graph
        self.base_patterns = base_patterns
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
        self.vertices = {i: [] for i in range(len(self.base_patterns))}
        ax_selected, n = np.zeros(len(self.base_patterns), dtype=int), 0

        sax_i, sax_got = self.get_signal_batch()

        for i in range(self.n_outputs):
            if self.firing_graph is not None:
                sax_fg = self.firing_graph.propagate(sax_i)[:, i]
            else:
                sax_fg = csc_matrix((self.batch_size, 1), dtype=bool)

            l_pattern_sub = [(j, struct) for j, struct in enumerate(self.base_patterns) if i in struct.O.nonzero()[1]]

            for j, pat in l_pattern_sub:

                l_pat_indices = list(set(pat.I.nonzero()[0]))
                sax_pat = pat.propagate(sax_i)[:, i]

                # selected random active
                ax_mask = sax_pat.multiply(sax_got[:, i]).toarray()[:, 0] & ~(sax_fg.toarray()[:, 0])
                n_sample = min(ax_mask.sum(), self.n_vertices)

                if n_sample == 0:
                    continue

                l_sampled_indices = np.random.randint(n_sample, size=n_sample)
                sax_samples = sax_i[ax_mask, :][l_sampled_indices, :]

                for sax_sample in sax_samples:
                    ax_indices = np.array([ind for ind in sax_sample.nonzero()[1] if ind not in l_pat_indices])
                    ax_mask = np.random.binomial(1, self.p_sample, len(ax_indices))

                    # Add sampled bits to list of output i vertices
                    if ax_mask.any():
                        self.vertices[j].extend([set(ax_indices[ax_mask > 0])])
                        ax_selected[j] += 1

        print("[Sampler]: Discriminative sampling has sampled {} vertices".format(ax_selected.sum()))

        return self
