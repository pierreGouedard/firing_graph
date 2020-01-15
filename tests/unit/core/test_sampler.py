# Global imports
import unittest
import numpy as np
from scipy.sparse import csc_matrix, hstack, lil_matrix

# Local import
from core.tools.imputers import ArrayImputer
from core.solver.sampler import SupervisedSampler
from core.data_structure.graph import FiringGraph


__maintainer__ = 'Pierre Gouedard'


class TestSampler(unittest.TestCase):
    def setUp(self):

        # Set core parameters
        self.ni, self.no, self.n_vertices, self.p_sample, self.weight = 100, 4, 2, 0.8, 10
        self.n_dead = 10
        # Generate random I / O
        self.sax_inputs = hstack(
            [csc_matrix(np.random.binomial(1, 0.5, (1000, self.ni - self.n_dead))), csc_matrix((1000, self.n_dead))]
        )
        self.sax_outputs = csc_matrix(np.random.binomial(1, 0.5, (1000, self.no)))

        # Set structures
        self.structures, self.vertices = [], {}
        for i in range(self.no):
            structures, vertices = generate_random_structures(i, 2, self.ni, self.no)
            self.structures.extend(structures)
            self.vertices[i] = vertices

    def test_create_firing_graph(self):
        """
        sampler test for first usage (initialisation)

        python -m unittest tests.unit.core.test_sampler.TestSampler.test_create_firing_graph

        """
        # Instantiate sampler
        imputer = init_imputer(self.sax_inputs, self.sax_outputs)
        sampler = SupervisedSampler(imputer, self.ni, self.no, self.p_sample, self.n_vertices)

        # Sample vertices
        sampler.generative_sampling()

        self.assertEqual(len(sampler.vertices), self.no)
        self.assertTrue(all([len(l_v) == self.n_vertices for _, l_v in sampler.vertices.items()]))

        firing_graph = sampler.build_firing_graph(self.weight)

        # Check structures
        self.assertEqual(len(sampler.structures), self.no)
        self.assertTrue(all([sampler.structures[i].Iw.shape[0] == self.ni for i in range(self.no)]))
        self.assertTrue(all([sampler.structures[i].Ow.shape[1] == self.no for i in range(self.no)]))
        self.assertTrue(all([sampler.structures[i].Cw.shape[0] == self.n_vertices + 1 for i in range(self.no)]))

        # Check dimension of the firing graph
        self.assertEqual(firing_graph.Iw.shape[0], self.ni)
        self.assertEqual(firing_graph.Ow.shape[1], self.no)
        self.assertEqual(firing_graph.Ow.shape[0], firing_graph.Iw.shape[1])
        self.assertEqual(firing_graph.Cw.shape[0], firing_graph.Iw.shape[1])
        self.assertEqual(firing_graph.Cw.shape[0], (self.n_vertices + 1) * self.no)
        self.assertEqual(firing_graph.depth, 3)

        # Check update masks
        self.assertTrue(not firing_graph.Im.toarray()[-self.n_dead:, :].any())
        self.assertTrue(firing_graph.Im.toarray()[:-self.n_dead, :].any())
        self.assertTrue(not firing_graph.Cm.toarray().any())
        self.assertTrue(not firing_graph.Om.toarray().any())
        self.assertTrue(not (firing_graph.Iw.toarray() > self.weight).any())

    def test_augment_firing_graph(self):
        """
        sampler test for first usage (initialisation)

        python -m unittest tests.unit.core.test_sampler.TestSampler.test_augment_firing_graph

        """
        # Instantiate sampler
        imputer = init_imputer(self.sax_inputs, self.sax_outputs)
        sampler = SupervisedSampler(
            imputer, self.ni, self.no, self.p_sample, self.n_vertices, structures=self.structures
        )

        # Sample vertices
        sampler.discriminative_sampling()

        # Check sampling
        self.assertEqual(len(sampler.vertices), len(self.structures))
        self.assertTrue(all([len(l_v) == self.n_vertices for _, l_v in sampler.vertices.items()]))

        firing_graph = sampler.build_firing_graph(self.weight)

        # Check structures
        self.assertEqual(len(sampler.structures), len(self.structures))
        self.assertTrue(all([sampler.structures[i].Iw.shape[0] == self.ni for i in range(len(self.structures))]))
        self.assertTrue(all([sampler.structures[i].Ow.shape[1] == self.no for i in range(len(self.structures))]))
        self.assertTrue(
            all([sampler.structures[i].Cw.shape[0] == self.n_vertices + 5 for i in range(len(self.structures))])
        )

        # Check dimension of the firing graph
        self.assertEqual(firing_graph.Iw.shape[0], self.ni)
        self.assertEqual(firing_graph.Ow.shape[1], self.no)
        self.assertEqual(firing_graph.Ow.shape[0], firing_graph.Iw.shape[1])
        self.assertEqual(firing_graph.Cw.shape[0], firing_graph.Iw.shape[1])
        self.assertEqual(firing_graph.Cw.shape[0], (self.n_vertices + 5) * len(self.structures))
        self.assertEqual(firing_graph.depth, 4)

        # Check update masks
        self.assertTrue(not any([fg.Im.toarray()[:, :3].any() for fg in sampler.structures]))
        self.assertTrue(not firing_graph.Cm.toarray().any())
        self.assertTrue(not firing_graph.Om.toarray().any())


def generate_random_structures(i, n_structure, n_inputs, n_outputs, n_pos=3, n_neg=2):
    """

    :param i:
    :param n_structure:
    :param n_inputs:
    :param n_outputs:
    :param n_pos:
    :param n_neg:
    :return:
    """
    l_vertices = [
        (
            np.random.choice(range(n_inputs), n_pos, replace=False),
            np.random.choice(range(n_inputs), n_neg, replace=False)
         ) for _ in range(n_structure)
    ]

    d_mask, ax_levels = {'I': np.zeros(n_inputs), 'C': np.zeros(3), 'O': np.zeros(n_outputs)}, np.ones(3)
    l_structures = []
    for bit_pos, bit_neg in l_vertices:

        # Init matrices
        sax_I, sax_C, sax_O = lil_matrix((n_inputs, 3)), lil_matrix((3, 3)), lil_matrix((3, n_outputs))

        # Set links
        sax_I[bit_pos, 0] = 1
        sax_I[bit_neg, 1] = 1
        sax_C[[0, 1], 2] = 1
        sax_O[2, i] = 1

        # Append structure
        l_structures.append(
            FiringGraph.from_matrices(sax_I, sax_C, sax_O, ax_levels.copy(), mask_vertices=d_mask.copy(), depth=3)
        )

    return l_structures, l_vertices


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