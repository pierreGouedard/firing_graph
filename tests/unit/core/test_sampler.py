# Global imports
import unittest
import numpy as np
from scipy.sparse import lil_matrix

# Local import
from firing_graph.tools.helpers.servers import ArrayServer
from firing_graph.tools.helpers.sampler import SupervisedSampler
from firing_graph.data_structure.graph import FiringGraph
from firing_graph.data_structure.utils import create_empty_matrices
__maintainer__ = 'Pierre Gouedard'


class TestSampler(unittest.TestCase):
    def setUp(self):

        # Set firing_graph parameters
        self.ni, self.no, self.weight = 100, 2, 10
        self.l_outputs = [[1, 11, 21, 31, 41, 51, 61, 71, 81, 91], [0, 25, 50, 75]]

        # Generate I / O
        self.sax_inputs = lil_matrix(np.eye(self.ni))
        self.sax_inputs[1, 0] = 1
        self.sax_outputs = lil_matrix((self.ni, self.no))
        for i, l_inds in enumerate(self.l_outputs):
            self.sax_outputs[self.l_outputs[i], i] = 1

        self.sax_inputs, self.sax_outputs = self.sax_inputs.tocsc(), self.sax_outputs.tocsc()

        # Set structures
        self.l_patterns, self.patterns = [[1, 21, 41, 61, 81], [0, 75]], []
        for i, l_inds in enumerate(self.l_patterns):
            self.patterns.append(generate_pattern(self.ni, self.no, i, self.l_patterns[i]))

        # Add another random pattern
        self.patterns.append(generate_pattern(self.ni, self.no, 0, [99]))

    def test_generative_sampling(self):
        """
        python -m unittest tests.unit.firing_graph.test_sampler.TestSampler.test_generative_sampling

        """
        # Instantiate samplers
        server = init_server(self.sax_inputs, self.sax_outputs)

        # Sampling with p_sample=0
        sampler = SupervisedSampler(server, self.ni, self.no, self.ni, 0, 5)
        sampler.generative_sampling()
        self.assertEqual(len(sampler.samples), self.no)
        for i in range(self.no):
            self.assertTrue(len(sampler.samples[i]) == 0)

        # Sampling with p_sample=1
        sampler = SupervisedSampler(server, self.ni, self.no, self.ni, 1, 5)
        sampler.generative_sampling()

        self.assertEqual(len(sampler.samples), self.no)
        if 0 in sampler.samples[0]:
            self.assertTrue(len(sampler.samples[0]) == 6)
            self.assertTrue(len(np.unique(sampler.samples[0])) == 6)
        else:
            self.assertTrue(len(sampler.samples[0]) == 5)
            self.assertTrue(len(np.unique(sampler.samples[0])) == 5)

        self.assertTrue(len(sampler.samples[1]) == len(self.l_outputs[1]))
        self.assertTrue(len(np.unique(sampler.samples[1])) == len(self.l_outputs[1]))

    def test_discriminative_sampling(self):
        """
        python -m unittest tests.unit.firing_graph.test_sampler.TestSampler.test_discriminative_sampling

        """
        # Instantiate samplers
        server = init_server(self.sax_inputs, self.sax_outputs)

        # Sampling with p_sample=0
        sampler = SupervisedSampler(server, self.ni, self.no, self.ni, 0, 5, patterns=self.patterns)
        sampler.discriminative_sampling()
        self.assertEqual(len(sampler.samples), len(self.patterns))
        for i in range(len(self.patterns)):
            self.assertTrue(len(sampler.samples[i]) == 0)

        # Sampling with p_sample=1
        sampler = SupervisedSampler(server, self.ni, self.no, self.ni, 1, 6,  patterns=self.patterns)
        sampler.discriminative_sampling()
        self.assertEqual(len(sampler.samples), len(self.patterns))
        self.assertTrue(len(sampler.samples[0]) == 1)
        self.assertTrue(sampler.samples[0][0] == 0)

        for i in range(1, len(self.patterns)):
            self.assertTrue(len(sampler.samples[i]) == 0)


def init_server(sax_input, sax_output):
    """

    :param sax_input:
    :param sax_output:
    :return:
    """

    # Create and init server
    server = ArrayServer(sax_input, sax_output)
    server.stream_features()

    return server


def generate_pattern(n_inputs, n_outputs, index_output, l_indices):
    """
    Create and return simple pattern (firing_graph).

    :param n_inputs:
    :param n_outputs:
    :param index_output:
    :param l_indices:
    :return:
    """
    # Initialize Matrices
    d_matrices = create_empty_matrices(n_inputs, n_outputs, 1)

    # Set level and matrices
    ax_levels = np.ones(1)
    d_matrices['Iw'][l_indices, 0] = 1
    d_matrices['Ow'][0, index_output] = 1

    # Add firing graph kwargs
    kwargs = {'ax_levels': ax_levels, 'matrices': d_matrices, 'project': 'SamplerTest', 'depth': 2}

    return FiringGraph(**kwargs)
