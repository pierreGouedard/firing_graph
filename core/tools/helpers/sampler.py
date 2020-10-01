# Global import
import numpy as np
import time

# local import


class SupervisedSampler(object):
    """
    This class implements a supervised samplers engine. It samples randomly input bit base on a concomitant activation
    of bit and output bit.
    """
    def __init__(self, server, n_batch, p_sample=0.8, pattern=None, verbose=0):
        """
        :param server: Serve input and  output activation.
        :type server: core.tools.servers.ArrayServer
        :param n_batch: Number of input grid state to read for sampling.
        :param p_sample: Input bit's sampling rate in [0., 1.]
        :param n_sampling: Number of sampling.
        :param patterns: firing graph.
        :type patterns: core.data_structure.firing_graph.FiringGraph.
        :param verbose: Control display.
        """
        # size of problem
        self.n_batch = n_batch

        # Sampling parameters
        self.p_sample = p_sample

        # Utils parameters
        self.verbose = verbose

        # Core attributes
        self.pattern = pattern
        self.samples = None
        self.server = server

    def get_signal_batch(self):
        """
        Read in a sparse matrices input and output grid state.
        :return: tuple of input and output sparse grid activation's matrices
        """
        sax_i = self.server.next_forward(n=self.n_batch).sax_data_forward
        sax_o = self.server.next_backward(n=self.n_batch).sax_data_backward

        # Apply mask if any
        if self.server.sax_mask_forward is not None:
            sax_o += sax_o.multiply(self.server.sax_mask_forward)
            sax_o.data %= 2
            sax_o.eliminate_zeros()

        return sax_i, sax_o

    def generative_sampling(self):
        """
        For each output bit, at n_sampling occasions, sample randomly activated input grid's bit when output bit is
        also active and the input grid does not activate firing_graph, if any set.

        :return: Current instance of the class.
        """
        # Init
        if self.server.pattern_backward is None:
            n_outputs = self.server.n_label
        else:
            n_outputs = self.server.pattern_backward.O.shape[1]

        self.samples, ax_selected, n = {i: [] for i in range(n_outputs)}, np.zeros(n_outputs, dtype=int), 0

        # Gte signal to sampled
        sax_i, sax_got = self.get_signal_batch()
        for i in range(self.server.n_label):

            # Selected random active
            ax_mask = sax_got.astype(bool)[:, i].toarray()[:, 0]
            n_sampling = min(ax_mask.sum(), 1)

            if n_sampling > 0:

                # Randomly Select grid state at target activations
                l_sampled_indices = np.random.choice(range(int(ax_mask.sum())), size=n_sampling, replace=False)
                sax_samples = sax_i[ax_mask, :][l_sampled_indices, :]

                # Sample active bits of selected grid state
                ax_indices = sax_samples.nonzero()[1]
                ax_mask = np.random.binomial(1, self.p_sample, len(ax_indices)).astype(bool)
                if ax_mask.any():
                    self.samples[i].extend(set(ax_indices[ax_mask]))
                    ax_selected[i] += 1

        print("[Sampler]: Generative sampling for {} targets.".format(ax_selected.sum()))
        return self

    def discriminative_sampling(self):
        """
        For each patterns, at n_sampling occasions, sample randomly activated input grid's bit when the input activate
        the pattern and the output bit linked to it and does not activate firing_graph, if any set.

        :return:
        """
        # Init
        tic = time.time()
        self.samples = {i: [] for i in range(len(self.pattern.partitions))}
        ax_selected, n = np.zeros(len(self.pattern.partitions), dtype=int), 0
        sax_i, sax_got = self.get_signal_batch()
        sax_data = self.pattern.propagate(sax_i)

        for i, p in enumerate(self.pattern.partitions):
            # Get current pattern input bit
            l_pat_indices = list(set(self.pattern.I[:, p['indices']].nonzero()[0]))
            sax_pat = sax_data[:, p['output_id']]

            # selected random active
            ax_mask = sax_pat.multiply(sax_got[:, p['output_id']]).astype(bool).toarray()[:, 0]
            n_sampling = min(ax_mask.sum(), 1)

            if n_sampling > 0:
                # Randomly Select grid state at target activations
                l_sampled_indices = np.random.choice(range(int(ax_mask.sum())), size=n_sampling, replace=False)
                sax_samples = sax_i[ax_mask, :][l_sampled_indices, :]

                # Sample active bits of selected grid state
                ax_indices = np.array([ind for ind in sax_samples.nonzero()[1] if ind not in l_pat_indices])
                ax_mask = np.random.binomial(1, self.p_sample, len(ax_indices)).astype(bool)
                if ax_mask.any():
                    self.samples[i].extend(set(ax_indices[ax_mask]))
                    ax_selected[i] += 1

        print("[Sampler]: Discriminative sampling for {} targets (out of {}). duration {}".format(
            ax_selected.sum(), len(self.pattern.partitions), time.time() - tic
        ))
        return self


class YalaSampler(object):
    """
    This class implements a supervised samplers engine. It samples randomly input bit base on a concomitant activation
    of bit and output bit.
    """
    def __init__(self, mapping_feature_input, p_sample=0.8, verbose=0):
        """
        :param server: Serve input and  output activation.
        :type server: core.tools.servers.ArrayServer
        :param n_batch: Number of input grid state to read for sampling.
        :param p_sample: Input bit's sampling rate in [0., 1.]
        :param n_sampling: Number of sampling.
        :param patterns: firing graph.
        :type patterns: core.data_structure.firing_graph.FiringGraph.
        :param verbose: Control display.
        """
        self.p_sample = p_sample
        self.mapping_feature_input = mapping_feature_input
        self.verbose = verbose
        self.samples = {}

    def sample(self, n):
        """
        For each output bit, at n_sampling occasions, sample randomly activated input grid's bit when output bit is
        also active and the input grid does not activate firing_graph, if any set.

        :return: Current instance of the class.
        """

        for i in range(n):
            # Sample feature
            ax_feature_mask = np.random.binomial(1, self.p_sample, self.mapping_feature_input.shape[1]).astype(bool)

            # Get input linked to sampled features
            ax_input_mask = self.mapping_feature_input.dot(np.diag(ax_feature_mask)).sum(axis=1) > 0

            # add inputes  to samples
            self.samples[i] = ax_input_mask.nonzero()[0]

        print("[Sampler]: {} sampling of features.".format(n))

        return self
