# Global imports
import unittest
import numpy as np
from scipy.sparse import csc_matrix

from core.tools.imputers.array import DoubleArrayImputer
from core.solver.sampler import Sampler

from core.tools.drivers.nmp import NumpyDriver

__maintainer__ = 'Pierre Gouedard'


class TestSampler(unittest.TestCase):
    def setUp(self):

        self.ni, self.no = 100, 2
        self.N = 100
        self.selected_bits = {0: {0, 1, 2, 3}, 1: {50, 51, 52}}

        self.pselected_bits, bits = {0: {0, 1, 2, 3}, 1: {50, 51, 52}}, set(range(self.ni))
        for i in range(self.no):
            self.pselected_bits[i] = self.pselected_bits[i].union(set(np.random.choice(list(bits), 30)))
            bits = bits.difference(self.pselected_bits[i])

        self.input = csc_matrix(np.random.binomial(1, 0.1, (100, self.ni)), dtype=int)
        self.output = csc_matrix(np.random.binomial(1, 0.1, (100, self.no)), dtype=int)

    def sampler_init(self):
        """
        sampler test for first usage (initialisation)

        python -m unittest tests.unit.core.sampler.TestSampler.sampler_init

        """
        # Create simple imputers and sampler
        imputer = init_imputer(self.input, self.output)
        sampler = Sampler((self.ni, self.no), self.N, imputer)

        # sample bits
        sampler.sample_supervised()

        # Make sure sampling is correct
        for i in sampler.preselect_bits[0]:
            self.assertTrue(self.input[:, i].transpose().dot(self.output[:, 0]) > 0)
        for i in sampler.preselect_bits[1]:
            self.assertTrue(self.input[:, i].transpose().dot(self.output[:, 1]) > 0)

        # build firing graph for drainer
        sampler.build_graph_multiple_output()

        # Test dim of Firing graph
        self.assertEqual(sampler.firing_graph.C.shape[0], 2)
        self.assertEqual(sampler.firing_graph.I.shape[0], self.ni)
        self.assertEqual(sampler.firing_graph.O.shape[1], self.no)

    def sampler_main(self):
        """
        sampler test for main usage of sampler (after initialisation)

        python -m unittest tests.unit.core.sampler.TestSampler.sampler_main

        """
        # Create simple imputers and sampler
        imputer = init_imputer(self.input, self.output)
        sampler = Sampler(
            (self.ni, self.no), self.N, imputer, selected_bits=self.selected_bits, preselected_bits=self.pselected_bits
        )

        # sample bits
        sampler.sample_supervised()

        # Make sure sampling is correct
        for i in sampler.preselect_bits[0]:
            self.assertTrue(self.input[:, i].transpose().dot(self.output[:, 0]) > 0)
        for i in sampler.preselect_bits[1]:
            self.assertTrue(self.input[:, i].transpose().dot(self.output[:, 1]) > 0)

        # build firing graph for drainer
        sampler.build_graph_multiple_output()

        # Test dim of Firing graph
        self.assertEqual(sampler.firing_graph.C.shape[0], 6)
        self.assertEqual(sampler.firing_graph.I.shape[0], self.ni)
        self.assertEqual(sampler.firing_graph.O.shape[1], self.no)

        # Test level of core vertices
        for i in range(self.no):
            l_core_vetices = sampler.core_vertices[i]
            self.assertTrue(
                list(sampler.firing_graph.levels[list(map(lambda x: int(x.split('_')[1]), l_core_vetices))]),
                [1, len(self.selected_bits[i]), 2]
            )


def init_imputer(ax_input, ax_output):
    # Create temporary directory for test
    driver = NumpyDriver()
    tmpdirin, tmpdirout = driver.TempDir('test_sampler', suffix='in', create=True), \
                          driver.TempDir('test_sampler', suffix='out', create=True)

    # Create I/O and save it into tmpdir files
    driver.write_file(ax_input, driver.join(tmpdirin.path, 'forward.npz'), is_sparse=True)
    driver.write_file(ax_output, driver.join(tmpdirin.path, 'backward.npz'), is_sparse=True)

    # Create and init imputers
    imputer = DoubleArrayImputer('test', tmpdirin.path, tmpdirout.path)
    imputer.read_raw_data('forward.npz', 'backward.npz')
    imputer.run_preprocessing()
    imputer.write_features('forward.npz', 'backward.npz')
    imputer.stream_features()

    tmpdirin.remove()
    tmpdirout.remove()

    return imputer