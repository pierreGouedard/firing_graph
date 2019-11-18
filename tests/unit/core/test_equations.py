# Global imports
import unittest
import numpy as np
from scipy.sparse import csc_matrix

from core.data_structure.utils import mat_from_tuples
from core.data_structure.graph import FiringGraph
from core.tools.imputers.array import DoubleArrayImputer
from core.solver.drainer import FiringGraphDrainer
from core.tools.drivers.nmp import NumpyDriver

__maintainer__ = 'Pierre Gouedard'


class TestEquations(unittest.TestCase):
    """
    Test forward:
        Graph:
              i0      i1
             /  \    / \
            /   \   /   \
           c0    c1      c2
            \    /        |
             \  /         |
              o0          o2

        Levels:
         c0 = 1, c1 = 2, c2 = 1

        Table Input -> Core:
        | i0 | i1 | -> | c0 | c1 | c2 |
        | 0  |  0 | -> | 0  |  0 | 0  |
        | 1  |  0 | -> | 1  |  0 | 0  |
        | 0  |  1 | -> | 0  |  0 | 1  |
        | 1  |  1 | -> | 1  |  1 | 1  |

        Table Core -> Output:
        | c0 | c1 | c2 | -> | o0 | o1 |
        | 0  |  0 | 0  | -> | 0  | 0  |
        | 1  |  0 | 0  | -> | 1  | 0  |
        | 0  |  0 | 1  | -> | 0  | 1  |
        | 1  |  1 | 1  | -> | 1  | 1  |

        Table Target Input -> Output:
        | i0 | i1 | -> | o0 | o1 |
        | 0  |  0 | -> | 0  |  0 |
        | 1  |  0 | -> | 0  |  0 |
        | 0  |  1 | -> | 0  |  1 |
        | 1  |  1 | -> | 1  |  1 |

    """
    def setUp(self):
        # Create a simple deep network (2 input vertices, 3 network vertices,, 2 output vertices)
        self.ni, self.nc, self.no, self.depth, self.weight, self.p, self.q, self.batch_size = 2, 3, 2, 2, 10, 3, 2, 4
        l_edges = [('input_0', 'core_0'), ('core_0', 'output_0')] +  \
                  [('input_0', 'core_1'), ('core_1', 'output_0')] + \
                  [('input_1', 'core_1')] + \
                  [('input_1', 'core_2'), ('core_2', 'output_1')]

        self.levels = np.array([1, 2, 1], dtype=int)
        self.sax_I, self.sax_C, self.sax_O = mat_from_tuples(self.ni, self.no, self.nc, l_edges, self.weight)
        self.mask_vertice_drain_a = {'I': np.ones(self.ni), 'C': np.ones(self.nc)}
        self.mask_vertice_drain_b = {'I': np.ones(self.ni), 'C': np.zeros(self.nc)}
        self.mask_vertice_drain_c = {'I': np.zeros(self.ni), 'C': np.ones(self.nc)}

        # Create firing graphs
        self.fga = FiringGraph.from_matrices(
            'test_equation_a', self.sax_I, self.sax_C, self.sax_O, self.levels,  self.mask_vertice_drain_a
        )

        self.fgb = FiringGraph.from_matrices(
            'test_equation_b', self.sax_I, self.sax_C, self.sax_O, self.levels,  self.mask_vertice_drain_b
        )

        self.fgc = FiringGraph.from_matrices(
            'test_equation_c', self.sax_I, self.sax_C, self.sax_O, self.levels,  self.mask_vertice_drain_c
        )

        # Create test signals
        self.input = csc_matrix([[0, 0], [1, 0], [0, 1], [1, 1]])
        self.output = csc_matrix([[0, 0], [0, 0], [0, 1], [1, 1]])

    def forward(self):
        """
        Very precise case on very simple graph to validate basics of drainer for forward equations
        python -m unittest tests.unit.core.equations.TestEquations.forward

        """

        # Create imputers and drainer
        imputer = init_imputer(self.input, self.output)
        drainer = FiringGraphDrainer(100, self.p, self.q, self.batch_size, self.fga, imputer)

        # Run for Two iteration and check forward signals are as expected
        drainer.run_iteration(True, False)
        drainer.iter += 1
        self.assertTrue((drainer.sax_c.toarray() == np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 1, 1]])).all())

        drainer.run_iteration(True, False)
        self.assertTrue((drainer.sax_o.toarray() == np.array([[0, 0], [1, 0], [0, 1], [2, 1]])).all())

    def backward(self):
        """
        Very precise case on very simple graph to validate basics of drainer for backward equations
        python -m unittest tests.unit.core.equations.TestEquations.backward

        """
        # Create imputers and drainer
        imputer = init_imputer(self.input, self.output)
        drainer = FiringGraphDrainer(100, self.p, self.q, self.batch_size, self.fga, imputer)

        # Run for Two iteration and check backward signals are as expected
        drainer.run_iteration(True, True)
        drainer.iter += 1
        drainer.forward_transmiting(True)
        drainer.forward_processing(True)

        # Output feedback
        drainer.backward_processing()

        # Build expected backward signals
        ax_expected = np.hstack(
            (np.zeros((self.no, self.batch_size)), np.array([[0, -3, 0, 2], [0, 0, 2, 2]]),
             np.zeros((self.no, 2 * self.batch_size)))
        )

        # Check correctness of feedback
        self.assertTrue((drainer.sax_ob.toarray() == ax_expected).all())

        # Core adjacency matrix update
        drainer.backward_transmiting()
        drainer.iter += 1

        # Build expected backward signals
        ax_O_expected = np.array([[self.q - self.p,  0.], [self.q,  0.], [0.,  2.*self.q]])
        ax_O_expected = ax_O_expected + (ax_O_expected != 0) * self.weight
        ax_O_track = np.array([[2,  0.], [1.,  0.], [0.,  2.]])

        # Check correctness of core structure updates
        self.assertTrue((drainer.firing_graph.Ow.toarray() == ax_O_expected).all())
        self.assertTrue((drainer.firing_graph.backward_firing['o'].toarray() == ax_O_track).all())

        drainer.forward_transmiting(True)
        drainer.forward_processing(True)
        drainer.backward_processing()
        drainer.backward_transmiting()

        # Build expected backward signals
        ax_I_expected = np.array([[self.q - self.p, self.q, 0], [0, self.q, 2. * self.q]])
        ax_I_expected = ax_I_expected + (ax_I_expected != 0) * self.weight
        ax_I_track = np.array([[2, 1, 0], [0, 1, 2]])

        # Check correctness of backward signals
        self.assertTrue((drainer.firing_graph.Iw.toarray() == ax_I_expected).all())
        self.assertTrue((drainer.firing_graph.backward_firing['i'].toarray() == ax_I_track).all())

    def drain_mask(self):
        """
        Very precise case on very simple graph to validate effectiveness of mask for draining
        python -m unittest tests.unit.core.equations.TestEquations.drain_mask

        """
        # Create imputers and drainer
        imputer = init_imputer(self.input, self.output)
        drainer = FiringGraphDrainer(100, self.p, self.q, self.batch_size, self.fgb, imputer)

        # Run for 1 epoch and check backward signals are as expected
        drainer.drain(1)

        # Build expected backward signals
        ax_I_expected = np.array([[self.q - self.p, self.q, 0], [0, self.q, 2. * self.q]])
        ax_I_expected = ax_I_expected + (ax_I_expected != 0) * self.weight
        ax_I_track = np.array([[2, 1, 0], [0, 1, 2]])

        # Check correctness of backward signals
        self.assertTrue((drainer.firing_graph.Iw.toarray() == ax_I_expected).all())
        self.assertTrue((drainer.firing_graph.backward_firing['i'].toarray() == ax_I_track).all())
        self.assertTrue((drainer.firing_graph.Ow.toarray() == self.fga.Ow.toarray()).all())

        # Create imputers and drainer
        imputer = init_imputer(self.input, self.output)
        drainer = FiringGraphDrainer(100, self.p, self.q, self.batch_size, self.fgc, imputer)

        # Run for 1 epoch and check backward signals are as expected
        drainer.drain(1)

        # Build expected backward signals
        ax_O_expected = np.array([[self.q - self.p,  0.], [self.q,  0.], [0.,  2. * self.q]])
        ax_O_expected = ax_O_expected + (ax_O_expected != 0) * self.weight
        ax_O_track = np.array([[2,  0.], [1.,  0.], [0.,  2.]])

        # Check correctness of core structure updates
        self.assertTrue((drainer.firing_graph.Ow.toarray() == ax_O_expected).all())
        self.assertTrue((drainer.firing_graph.backward_firing['o'].toarray() == ax_O_track).all())
        self.assertTrue((drainer.firing_graph.Iw.toarray() == self.fga.Iw.toarray()).all())


def init_imputer(ax_input, ax_output):
    # Create temporary directory for test
    driver = NumpyDriver()
    tmpdirin, tmpdirout = driver.TempDir('test_equation', suffix='in', create=True), \
                          driver.TempDir('test_equation', suffix='out', create=True)

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