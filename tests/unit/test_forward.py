# Global imports
import unittest
import numpy as np
from scipy.sparse import csr_matrix, eye

# Local import
from firing_graph.graph import FiringGraph, create_empty_matrices
from firing_graph.linalg.forward import ftc, fast_partitioned_ftc, fpo, fto
from .data import d_graphs, d_signals


class TestForward(unittest.TestCase):
    """
    """
    def setUp(self):

        # FTC test graph 1 convex case (input only)
        d_matrices = self.build_matrices(d_graphs[1]['I'], d_graphs[1]['C'], d_graphs[1]['O'])
        self.signal1 = self.build_signals(d_signals[1])
        self.fg1 = FiringGraph('test1', np.array(d_graphs[1]['levels']), d_matrices)

        # FTC test graph 2 non convex case (input only)
        d_matrices = self.build_matrices(d_graphs[2]['I'], d_graphs[2]['C'], d_graphs[2]['O'])
        self.signal2 = self.build_signals(d_signals[2])
        self.fg2 = FiringGraph('test2', np.array(d_graphs[2]['levels']), d_matrices)

        # FTC test graph 3 of mixte convex case (input + core)
        d_matrices = self.build_matrices(d_graphs[3]['I'], d_graphs[3]['C'], d_graphs[3]['O'])
        self.signal3 = self.build_signals(d_signals[3])
        self.fg3 = FiringGraph('test2', np.array(d_graphs[3]['levels']), d_matrices)

        # Partitioned FTC test graph1
        d_matrices = self.build_matrices(d_graphs[4]['I'], d_graphs[4]['C'], d_graphs[4]['O'])
        self.signal4 = self.build_signals(d_signals[4])
        self.fg4 = FiringGraph(
            'test4', np.array(d_graphs[4]['levels']), d_matrices, meta=d_graphs[4]['input_meta']
        )

        # FPO tests
        self.signal5 = d_signals[5]

        # FTO tests
        self.signal6 = d_signals[6]

    @staticmethod
    def build_matrices(l_inputs, l_cores, l_outputs):

        # init matrices
        n_outputs = len(l_outputs[0]) if l_outputs else len(l_inputs)
        d_matrices = create_empty_matrices(len(l_inputs[0]), len(l_inputs), n_outputs, write_mode=False)

        # Set matrices
        d_matrices['Iw'] = csr_matrix(l_inputs).T.tocsr()
        d_matrices['Im'] = csr_matrix(l_inputs).T.tocsr()
        d_matrices['O'] = eye(2, format='csr', dtype=bool)

        if l_cores is not None:
            d_matrices['C'] = csr_matrix(l_cores, dtype=bool).T
        if l_outputs is not None:
            d_matrices['Ow'] = csr_matrix(l_outputs, dtype=bool)

        return d_matrices

    @staticmethod
    def build_signals(d_signals):
        return {
            'i': csr_matrix(d_signals['input'], dtype=bool),
            'e': np.array(d_signals['expected'], dtype=bool),
            'c': csr_matrix(d_signals['core'], dtype=bool),
        }

    def test_ftc(self):
        """
        python -m unittest tests.unit.test_forward.TestForward.test_ftc
        """

        # Test1 run
        sax_res = ftc(self.fg1.I, self.signal1['i'], self.fg1.C, self.signal1['c'], self.fg1.levels)

        # Test1 validation
        self.assertTrue((sax_res.A == self.signal1['e']).all())

        # Test2 run
        sax_res = ftc(self.fg2.I, self.signal2['i'], self.fg2.C, self.signal2['c'], self.fg2.levels)

        # Test2 validation
        self.assertTrue((sax_res.A == self.signal2['e']).all())

        # Test3 run
        sax_res = ftc(self.fg3.I, self.signal3['i'], self.fg3.C, self.signal3['c'], self.fg3.levels)

        # Test3 validation
        self.assertTrue((sax_res.A == self.signal3['e']).all())

    def test_partitioned_ftc(self):
        """
        python -m unittest tests.unit.test_forward.TestForward.test_partitioned_ftc
        """
        # Test4 run
        sax_res = fast_partitioned_ftc(self.fg4.meta, self.signal4['i'], self.fg4.levels)

        # Test4 validation
        self.assertTrue((sax_res.A == self.signal4['e']).all())

    def test_fpo(self):
        """
        python -m unittest tests.unit.test_forward.TestForward.test_fpo

        """
        sax_res = fpo(self.signal5['out'], self.signal5['got'], self.signal5['p'], self.signal5['r'])
        # Test4 validation
        self.assertTrue((sax_res.A == self.signal5['expected']).all())

    def test_fto(self):
        """
        python -m unittest tests.unit.test_forward.TestForward.test_fto

        """
        # 1 job
        sax_res = fto(self.signal6['O'].T, self.signal6['c'])
        self.assertTrue((sax_res.A == self.signal6['expected']).all())

        # Multiprocessing
        sax_res = fto(self.signal6['O'].T, self.signal6['c'], njobs=2)
        self.assertTrue((sax_res.A == self.signal6['expected']).all())
