# Global imports
import unittest
import numpy as np
from scipy.sparse import csr_matrix, eye

# Local import
from firing_graph.graph import FiringGraph, create_empty_matrices
from .data import d_graphs, d_signals


class TestFiringGraph(unittest.TestCase):
    """
    """
    def setUp(self):

        # test FG depth 2 not partitioned
        d_matrices = self.build_matrices(d_graphs[7]['I'])
        self.signal1 = self.build_signals(d_signals[9])
        self.fg1 = FiringGraph('test1', np.array(d_graphs[7]['levels']), d_matrices)

        # Test FG depth 2 partitioned
        d_matrices = self.build_matrices(d_graphs[8]['I'])
        self.signal2 = self.build_signals(d_signals[10])
        self.fg2 = FiringGraph(
            'test2', np.array(d_graphs[8]['levels']), d_matrices, input_partitions=d_graphs[8]['input_partitions']
        )

        # Todo: FG depth > 2

    @staticmethod
    def build_matrices(l_inputs):
        # init matrices
        d_matrices = create_empty_matrices(len(l_inputs[0]), len(l_inputs), len(l_inputs), write_mode=False)

        # Set matrices
        d_matrices['Iw'] = csr_matrix(l_inputs).T.tocsr()
        d_matrices['Im'] = csr_matrix(l_inputs, dtype=bool).T.tocsr()
        d_matrices['O'] = eye(len(l_inputs), format='csr', dtype=bool)

        return d_matrices

    @staticmethod
    def build_signals(d_signals):
        return {
            'i': csr_matrix(d_signals['input'], dtype=bool),
            'e': np.array(d_signals['expected'], dtype=bool),
        }

    def test_propagate(self):
        """
        python -m unittest tests.unit.test_fg.TestFiringGraph.test_propagate
        """
        # Test 1 not partitioned graph
        sax_res = self.fg1.seq_propagate(self.signal1['i'])

        # Validation
        self.assertTrue((self.signal1['e'] == sax_res.A).all())

        # Test 2 partitioned graph
        sax_res = self.fg2.seq_propagate(self.signal2['i'])

        # Validation
        self.assertTrue((self.signal2['e'] == sax_res.A).all())

