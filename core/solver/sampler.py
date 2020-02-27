# Global import
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, vstack

# local import


class SupervisedSampler(object):
    """
    This class implements a supervised sampler engine. It samples randomly input bit base on a concomitant activation
    of bit and output bit.
    """
    def __init__(self, server, n_inputs, n_outputs, n_batch, p_sample=0.8, n_samples=10, l0=1, firing_graph=None,
                 patterns=None, verbose=0):
        """
        :param server: Serve input and  output activation.
        :type server: core.tools.servers.ArrayServer
        :param n_inputs: Number of input bit.
        :param n_outputs: Number of input bit.
        :param n_batch: Number of input grid state to read for sampling.
        :param p_sample: Input bit's sampling rate in [0., 1.]
        :param n_samples: Number of sampling.
        :param firing_graph: Firing graph.
        :type firing_graph: core.data_structure.firing_graph.FiringGraph
        :param patterns: firing graph.
        :type patterns: core.data_structure.firing_graph.FiringGraph.
        :param verbose: Control display.
        """
        # size of problem
        self.n_inputs, self.n_outputs, self.n_batch = n_inputs, n_outputs, n_batch

        # Sampling parameters
        self.p_sample, self.n_samples = p_sample, n_samples

        # Utils parameters
        self.verbose = verbose

        # Structure  parameter
        self.l0 = l0

        # Core attributes
        self.firing_graph = firing_graph
        self.patterns = patterns
        self.vertices = None
        self.server = server

    def get_signal_batch(self):
        """
        Read in a sparse matrices input and output grid state.
        :return: tuple of input and output sparse grid activation's matrices
        """
        sax_i = self.server.next_forward(n=self.n_batch)
        sax_o = self.server.next_backward(n=self.n_batch)

        return sax_i, sax_o

    def generative_sampling(self):
        """
        For each output bit, at n_samples occasions, sample randomly activated input grid's bit when output bit is
        also active and the input grid does not activate firing_graph, if any set.

        :return: Current instance of the class.
        """
        # Init
        self.vertices = {i: [] for i in range(self.n_outputs)}
        ax_selected, n = np.zeros(self.n_outputs, dtype=int), 0

        sax_i, sax_got = self.get_signal_batch()

        for i in range(self.n_outputs):
            # selected random active
            ax_mask = sax_got[:, i].toarray()[:, 0]
            n_sample = min(ax_mask.sum(), self.n_samples)

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
        """
        For each patterns, at n_samples occasions, sample randomly activated input grid's bit when the input activate
        the pattern and the output bit linked to it and does not activate firing_graph, if any set.

        :return:
        """
        # Init
        self.vertices = {i: [] for i in range(len(self.patterns))}
        ax_selected, n = np.zeros(len(self.patterns), dtype=int), 0

        sax_i, sax_got = self.get_signal_batch()

        for i in range(self.n_outputs):
            l_pattern_sub = [(j, struct) for j, struct in enumerate(self.patterns) if i in struct.O.nonzero()[1]]

            for j, pat in l_pattern_sub:

                l_pat_indices = list(set(pat.I.nonzero()[0]))
                sax_pat = pat.propagate(sax_i)[:, i]

                # selected random active
                ax_mask = sax_pat.multiply(sax_got[:, i]).toarray()[:, 0]
                n_sample = min(ax_mask.sum(), self.n_samples)

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
