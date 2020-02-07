# Global imports
import unittest
import numpy as np
from scipy.sparse import csc_matrix, hstack

# Local import
from core.tools.imputers import ArrayImputer
from core.solver.sampler import SupervisedSampler
from core.data_structure.graph import FiringGraph
from core.data_structure.utils import create_empty_matrices
__maintainer__ = 'Pierre Gouedard'


class TestSampler(unittest.TestCase):
    def setUp(self):

        # Set core parameters
        self.ni, self.no, self.n_vertices, self.p_sample, self.weight = 100, 4, 2, 0.8, 10
        self.n_dead, self.precision, self.batch_size = 10, 0.5, 100

        # Generate random I / O
        self.sax_inputs = hstack(
            [csc_matrix(np.random.binomial(1, 0.7, (1000, self.ni - self.n_dead))), csc_matrix((1000, self.n_dead))]
        )
        self.sax_outputs = csc_matrix(np.random.binomial(1, 0.5, (1000, self.no)))

        # Set structures
        self.base_patterns, self.indices = [], []
        for i in range(self.no):
            self.indices.append(np.random.randint(0, self.ni - self.n_dead, 5))
            self.base_patterns.append(
                generate_pattern(self.ni, self.no, i, self.indices[-1])
            )

    def test_generative_sampling(self):
        """
        python -m unittest tests.unit.core.test_sampler.TestSampler.test_generative_sampling

        """
        # Instantiate sampler
        imputer = init_imputer(self.sax_inputs, self.sax_outputs)
        sampler = SupervisedSampler(imputer, self.ni, self.no, self.batch_size, self.p_sample, self.n_vertices)

        # Sample vertices
        sampler.generative_sampling()
        self.assertEqual(len(sampler.vertices), self.no)
        self.assertTrue(all([len(l_v) == self.n_vertices for _, l_v in sampler.vertices.items()]))

    def test_discriminative_sampling(self):
        """
        python -m unittest tests.unit.core.test_sampler.TestSampler.test_discriminative_sampling

        """
        # Instantiate sampler
        imputer = init_imputer(self.sax_inputs, self.sax_outputs)
        sampler = SupervisedSampler(
            imputer, self.ni, self.no, self.batch_size, self.p_sample, self.n_vertices, base_patterns=self.base_patterns
        )

        # Sample vertices
        sampler.discriminative_sampling()

        # Check sampling
        self.assertEqual(len(sampler.vertices), len(self.base_patterns))
        self.assertTrue(all([len(l_v) == self.n_vertices for _, l_v in sampler.vertices.items()]))

        # Make sure that base_pattern corresponding to sampled overlap with indices sampled
        for i, pattern in enumerate(self.base_patterns):
            ax_pattern = np.multiply(
                pattern.propagate(self.sax_inputs).toarray()[:, i],
                self.sax_outputs.toarray()[:, i]
            )
            self.assertTrue(not (ax_pattern.dot(self.sax_inputs[:, -self.n_dead].toarray()) > 0).any())
            for j, l_ind in enumerate(sampler.vertices[i]):
                ax_signals_sampled = self.sax_inputs[:, list(l_ind)].toarray()
                self.assertTrue((ax_pattern.dot(ax_signals_sampled) > 0).all())


def init_imputer(sax_input, sax_output):
    """

    :param sax_input:
    :param sax_output:
    :return:
    """

    # Create and init imputers
    imputer = ArrayImputer(sax_input, sax_output)
    imputer.stream_features()

    return imputer


def generate_pattern(n_inputs, n_outputs, index_output, l_indices):

    # Initialize Matrices
    d_matrices = create_empty_matrices(n_inputs, n_outputs, 1)

    # Set level and matrices
    ax_levels = np.ones(1) * (len(l_indices) - 1)
    d_matrices['Iw'][l_indices, 0] = 1
    d_matrices['Ow'][0, index_output] = 1

    # Add firing graph kwargs
    kwargs = {'ax_levels': ax_levels, 'matrices': d_matrices, 'project': 'SamplerTest', 'depth': 2}

    return FiringGraph(**kwargs)
