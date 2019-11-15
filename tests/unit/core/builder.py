# Global imports
import unittest
import numpy as np

# Local import
from core.data_structure.utils import mat_from_tuples
from core.data_structure.graph import FiringGraph


class TestBuilder(unittest.TestCase):
    def setUp(self):

        # Create a simple deep network (2 input nodes, 3 network nodes, 3 output nodes)
        self.ni, self.nc, self.no = 2, 5, 3
        self.l_edges = [('input_0', 'core_0'), ('core_0', 'core_1'), ('core_0', 'core_2'),
                        ('core_1', 'output_0'), ('core_2', 'output_1')] + \
                       [('input_0', 'core_3'), ('core_3', 'core_2')] + \
                       [('input_1', 'core_4'), ('core_4', 'output_2')] + \
                       [('input_1', 'core_2')]

        self.weight = 10
        self.ax_levels = [1, 1, 1, 1, 1]
        self.mask_vertice_drain = {'I': np.zeros(self.ni), 'C': np.zeros(self.nc), 'O': np.zeros(self.no)}

        # Get matrices for building test
        self.sax_I, self.sax_C, self.sax_O = mat_from_tuples(self.ni, self.no, self.nc, self.l_edges, self.weight)

    def building_graph(self):
        """
        Test basic graph building
        python -m unittest tests.unit.core.builder.TestBuilder.building_graph

        """

        # Update mask vertice drainer
        self.mask_vertice_drain['I'][0] = 1
        self.mask_vertice_drain['C'][3] = 1
        self.mask_vertice_drain['C'][2] = 1

        firing_graph = FiringGraph.from_matrices(
            'test_build', self.sax_I, self.sax_C, self.sax_O, self.ax_levels,  self.mask_vertice_drain
        )

        # Assert adjacency matrices are correct
        self.assertTrue(firing_graph.I[0, 3] and firing_graph.Iw[0, 3] == self.weight)
        self.assertTrue(firing_graph.C[3, 2] and firing_graph.Cw[3, 2] == self.weight)
        self.assertTrue(firing_graph.O[4, 2] and firing_graph.Ow[4, 2] == self.weight)

        # Assert mask for drainer are correct
        self.assertTrue(firing_graph.Im[1, :].nnz == 0 and firing_graph.Im[0, :].nnz == firing_graph.I[0, :].nnz)
        self.assertTrue(firing_graph.Cm[[0, 1, 2, 4], :].nnz == 0 and firing_graph.Cm[3, :].nnz == firing_graph.C[3, :].nnz)
        self.assertTrue(firing_graph.Om[[0, 1, 3, 4], :].nnz == 0 and firing_graph.Om[2, :].nnz == firing_graph.O[2, :].nnz)



