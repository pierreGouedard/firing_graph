# Global imports
import unittest
import numpy as np
from scipy.sparse import csr_matrix, eye

# Local import
from firing_graph.graph import FiringGraph, create_empty_matrices
from firing_graph.drainer import FiringGraphDrainer
from firing_graph.servers import ArrayServer
from .data import d_graphs, d_signals


class TestDrainer(unittest.TestCase):
    """
    """
    def setUp(self):
        d_matrices = self.build_matrices(d_graphs[9]['Iw'], d_graphs[9]['Im'])
        server = ArrayServer(
            csr_matrix(d_signals[11]['input'], dtype=bool), csr_matrix(d_signals[11]['got'], dtype=bool)
        )
        fg = FiringGraph(
            'test', np.array(d_graphs[9]['levels']), d_matrices, input_partitions=d_graphs[4]['input_partitions']
        )
        self.signal1 = self.build_signals(d_signals[11])
        self.drainer = FiringGraphDrainer(
            fg, server, d_signals[11]['bs'], penalties=d_signals[11]['p'], rewards=d_signals[11]['r']
        )

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
            'o': csr_matrix(d_signals['out'], dtype=bool),
            'cb': csr_matrix(d_signals['cb'], dtype=np.int32),
            'e': d_signals['expected']
        }

    def test_drain(self):
        """
        python -m unittest tests.unit.test_drainer.TestDrainer.test_drain
        """
        self.drainer.drain()
        self.assertTrue((np.array(self.signal1['e']['Iw']).T == self.drainer.firing_graph.matrices['Iw']).all())
        self.assertTrue((np.array(self.signal1['e']['Ic']).T == self.drainer.firing_graph.backward_firing).all())
        self.assertTrue(((np.array(self.signal1['e']['Iw']).T > 0) == self.drainer.firing_graph.I).all())
