# Global imports
import unittest
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix

# Local import
from core.data_structure.utils import mat_from_tuples
from core.data_structure.graph import FiringGraph


class TestBuilder(unittest.TestCase):
    def setUp(self):

        # Create a simple deep network (2 input nodes, 3 network nodes, 3 output nodes)
        self.ni, self.nc, self.no, self.weight = 2, 5, 3, 10

        # Set edges
        self.l_edges = [('input_0', 'core_0'), ('core_0', 'core_1'), ('core_0', 'core_2'),
                        ('core_1', 'output_0'), ('core_2', 'output_1')] + \
                       [('input_0', 'core_3'), ('core_3', 'core_2')] + \
                       [('input_1', 'core_4'), ('core_4', 'output_2')] + \
                       [('input_1', 'core_2')]

        self.l_edges_2 = [('input_0', 'core_0'), ('core_0', 'core_1'), ('core_0', 'core_2'),
                          ('core_1', 'output_0'), ('core_2', 'output_1')] + \
                         [('input_1', 'core_3'), ('core_3', 'core_2')]

        # Set levels
        self.ax_levels = np.array([1, 1, 1, 1, 1])
        self.ax_levels_2 = np.array([1, 1, 2, 1])

        # Set input and expected output of propagation through firing graph
        self.sax_i = csc_matrix(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        self.sax_o = csc_matrix(np.array([[0, 0], [0, 0], [1, 0], [1, 1]]))

        # Disable edge update
        self.mask_vertice_drain = {'I': np.zeros(self.ni), 'C': np.zeros(self.nc), 'O': np.zeros(self.no)}
        self.mask_vertice_drain_2 = {'I': np.zeros(self.ni), 'C': np.zeros(self.nc - 1), 'O': np.zeros(self.no - 1)}

        # Get matrices for building test
        self.sax_I, self.sax_C, self.sax_O = mat_from_tuples(self.ni, self.no, self.nc, self.l_edges, self.weight)

    def test_building_graph(self):
        """
        Test basic graph building
        python -m unittest tests.unit.core.test_builder.TestBuilder.test_building_graph

        """

        # Update mask vertice drainer
        self.mask_vertice_drain['I'][0] = 1
        self.mask_vertice_drain['C'][3] = 1
        self.mask_vertice_drain['C'][2] = 1

        firing_graph = FiringGraph.from_matrices(
            self.sax_I, self.sax_C, self.sax_O, self.ax_levels,  mask_vertices=self.mask_vertice_drain, depth=3
        )

        # Assert matrices dtypes and format are correct
        self.assertTrue(isinstance(firing_graph.Iw, csc_matrix) and firing_graph.Iw.dtype.type == np.int32)
        self.assertTrue(isinstance(firing_graph.Cw, csc_matrix) and firing_graph.Cw.dtype.type == np.int32)
        self.assertTrue(isinstance(firing_graph.Ow, csc_matrix) and firing_graph.Ow.dtype.type == np.int32)
        self.assertTrue(isinstance(firing_graph.Im, lil_matrix) and firing_graph.Im.dtype.type == np.bool_)
        self.assertTrue(isinstance(firing_graph.Cm, lil_matrix) and firing_graph.Cm.dtype.type == np.bool_)
        self.assertTrue(isinstance(firing_graph.Om, lil_matrix) and firing_graph.Om.dtype.type == np.bool_)

        # Assert tracking dtypes and format are correct
        self.assertTrue(isinstance(firing_graph.backward_firing['i'], csc_matrix))
        self.assertTrue(isinstance(firing_graph.backward_firing['c'], csc_matrix))
        self.assertTrue(isinstance(firing_graph.backward_firing['o'], csc_matrix))
        self.assertTrue(all([sax_bf.dtype.type == np.uint32 for sax_bf in firing_graph.backward_firing.values()]))

        # Assert adjacency matrices are correct
        self.assertTrue(firing_graph.I[0, 3] and firing_graph.Iw[0, 3] == self.weight)
        self.assertTrue(firing_graph.C[3, 2] and firing_graph.Cw[3, 2] == self.weight)
        self.assertTrue(firing_graph.O[4, 2] and firing_graph.Ow[4, 2] == self.weight)

        # Assert mask for drainer are correct
        self.assertTrue(firing_graph.Im[1, :].nnz == 0 and firing_graph.Im[0, :].nnz == firing_graph.I[0, :].nnz)
        self.assertTrue(firing_graph.Cm[[0, 1, 2, 4], :].nnz == 0 and firing_graph.Cm[3, :].nnz == firing_graph.C[3, :].nnz)
        self.assertTrue(firing_graph.Om[[0, 1, 3, 4], :].nnz == 0 and firing_graph.Om[2, :].nnz == firing_graph.O[2, :].nnz)

    def test_propagate(self):
        """
        Test basic graph building
        python -m unittest tests.unit.core.test_builder.TestBuilder.test_propagate

        """

        sax_I, sax_C, sax_O = mat_from_tuples(self.ni, self.no - 1, self.nc -1, self.l_edges_2, self.weight)

        firing_graph = FiringGraph.from_matrices(
            sax_I, sax_C, sax_O, self.ax_levels_2,  mask_vertices=self.mask_vertice_drain_2, depth=3
        )

        sax_o = firing_graph.propagate(self.sax_i)

        # Assert result is as expected
        self.assertTrue((sax_o.toarray() == self.sax_o.toarray()).all())
