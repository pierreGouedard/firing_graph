# Global imports
import unittest
import numpy as np
from scipy.sparse import csr_matrix, eye
import time

# Local import
from firing_graph.graph import FiringGraph, create_empty_matrices
from firing_graph.linalg.backward import bui
from .data import d_graphs, d_signals


class TestBackward(unittest.TestCase):
    """
    """
    def setUp(self):
        # BUI without specific mask
        d_matrices = self.build_matrices(d_graphs[5]['Iw'], d_graphs[5]['Im'])
        self.signal1 = self.build_signals(d_signals[7])
        self.fg1 = FiringGraph('test1', np.array(d_graphs[5]['levels']), d_matrices)

        # BUI test with specific mask
        d_matrices = self.build_matrices(d_graphs[6]['Iw'], d_graphs[6]['Im'])
        self.signal2 = self.build_signals(d_signals[8])
        self.fg2 = FiringGraph('test2', np.array(d_graphs[6]['levels']), d_matrices)

    @staticmethod
    def build_matrices(l_inputs, l_mask_inputs):
        # init matrices
        n_outputs = len(l_inputs)
        d_matrices = create_empty_matrices(len(l_inputs[0]), len(l_inputs), n_outputs, write_mode=False)

        # Set matrices
        d_matrices['Iw'] = csr_matrix(l_inputs).T.tocsr()
        d_matrices['Im'] = csr_matrix(l_mask_inputs, dtype=bool).T.tocsr()
        d_matrices['O'] = eye(n_outputs, format='csr', dtype=bool)

        return d_matrices

    @staticmethod
    def build_signals(d_signals):
        return {
            'i': csr_matrix(d_signals['input'], dtype=bool),
            'cb': csr_matrix(d_signals['core_backward'], dtype=np.int32),
            'e': d_signals['expected']
        }

    def test_bui(self):
        """
        python -m unittest tests.unit.test_backward.TestBackward.test_bui
        """
        # Without mask
        sax_res = bui(self.signal1['cb'], self.signal1['i'], self.fg1)

        # validation
        self.assertTrue((np.array(self.signal1['e']['Iw']).T == self.fg1.matrices['Iw']).all())
        self.assertTrue((np.array(self.signal1['e']['Ic']).T == sax_res).all())

        self.fg1.refresh_matrices()
        self.assertTrue(((np.array(self.signal1['e']['Iw']).T > 0) == self.fg1.I).all())

        # With mask
        sax_res = bui(self.signal2['cb'], self.signal2['i'], self.fg2)

        # Validation
        self.assertTrue((np.array(self.signal2['e']['Iw']).T == self.fg2.matrices['Iw']).all())
        self.assertTrue((np.array(self.signal2['e']['Ic']).T == sax_res).all())

        self.fg2.refresh_matrices()
        self.assertTrue(((np.array(self.signal2['e']['Iw']).T > 0) == self.fg2.I).all())

    @staticmethod
    def performance_bui():
        """
        python -m unittest tests.unit.test_backward.TestBackward.performance_bui
        """
        # Generate fake data
        sax_i = csr_matrix(np.random.binomial(1, 0.1, (100000, 100)), dtype=np.bool)
        sax_cb = csr_matrix(np.random.binomial(10, 0.01, (100000, 500)), dtype=np.bool)
        sax_I = csr_matrix(np.random.binomial(1, 0.1, (100, 500)), dtype=bool)

        class MockFg:
            I = sax_I.copy()
            Im = sax_I.copy()
            matrices = {'Iw': csr_matrix(sax_I.shape, dtype=np.int32)}

        # Show average performance
        l_times = []
        for i in range(50):
            t0 = time.time()
            _ = bui(sax_cb, sax_i, MockFg(), njobs=4)
            l_times.append(time.time() - t0)
            print(l_times[-1])

        print(f'Average performance {np.mean(l_times)}')


