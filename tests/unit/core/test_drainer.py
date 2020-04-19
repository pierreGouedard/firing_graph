# Global imports
import unittest
from scipy.sparse import csc_matrix, vstack, lil_matrix
import numpy as np

# Local import
from core.data_structure.utils import gather_matrices, create_empty_matrices
from core.tools.helpers.servers import ArrayServer
from core.solver.drainer import FiringGraphDrainer
from core.data_structure.graph import FiringGraph
from utils.interactive_plots import plot_graph
from utils.patterns import AndPattern2 as ap2, AndPattern3 as ap3

__maintainer__ = 'Pierre Gouedard'


class TestDrainer(unittest.TestCase):
    def setUp(self):

        # enable, disable visual inspection of graph
        self.visual = False

        # Create And pattern of depth 2 /!\ Do not change those parameter for the test /!\
        self.n, self.ni, self.no, self.w0, self.t_mask = 100, 10, 2, 10, 5
        self.ap2 = ap2(self.ni, self.no, w=self.w0, seed=1234)
        self.ap2_fg = self.ap2.build_graph_pattern_init()

        # Create And pattern of depth 3
        self.ni, self.no, self.n_selected = 15, 2, 3
        self.ap3 = ap3(self.ni, self.no, n_selected=self.n_selected, w=self.w0, seed=1234)
        self.ap3_fg = self.ap3.build_graph_pattern_init()

        # Create simple pattern to test multi output
        self.sax_forward = vstack([csc_matrix(np.eye(4)) for _ in range(10)])
        self.sax_backward = lil_matrix(self.sax_forward.shape)
        self.sax_backward[:, :2] = 1
        self.p, self.r = np.array([1, 1, 2, 1]), np.array([1, 2, 1, 1])
        self.weight = 100

        # Create patterns to test functionality of forward and  backward patterns in server
        d_matrices = create_empty_matrices(self.sax_forward.shape[1], self.sax_backward.shape[1], 4)
        for i in range(4):
            d_matrices['Iw'][i, i], d_matrices['Ow'][i, i], d_matrices['Im'][:, :] = self.weight, 1, True

        self.fg = FiringGraph('test_server', np.ones(4), d_matrices, depth=2)

    def test_time_mask(self):
        """
        Test the well functioning of mask on backward updates
        python -m unittest tests.unit.core.test_drainer.TestDrainer.test_time_mask

        """

        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.ap2.generate_io_sequence(1000, seed=1234)
        server = create_server(csc_matrix(ax_input), csc_matrix(ax_output))

        # Create drainer
        drainer = FiringGraphDrainer(
            self.ap2_fg, server, t=self.t_mask, p=np.ones(self.ap2.no), r=np.ones(self.ap2.no), batch_size=1
        )
        drainer.drain(100)

        # Get matrice of the graph
        fg, fg_final, fg_init = drainer.firing_graph, self.ap2.build_graph_pattern_final(), \
                                self.ap2.build_graph_pattern_init()
        I, I_init, I_final = fg.Iw.toarray(), fg_init.Iw.toarray(), fg_final.Iw.toarray()

        # Assert mask are working (no more than self.t_mask structure update
        self.assertTrue((I[I >= self.w0] == I_final[I_final > 0] + self.t_mask).all())
        self.assertTrue((I[(0 < I) & (I <= self.w0)] == I_init[~(I_final > 0) & (I_init > 0)] - self.t_mask).all())

    def test_andpattern(self):
        """
        Test And Pattern of depth 3
        python -m unittest tests.unit.core.test_drainer.TestDrainer.test_andpattern

        """
        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.ap3.generate_io_sequence(1000, seed=1234)
        server = create_server(csc_matrix(ax_input), csc_matrix(ax_output))

        # Create drainer
        drainer = FiringGraphDrainer(
            self.ap3_fg, server, t=1000, p=np.ones(self.ap3.no), r=np.ones(self.ap3.no), batch_size=1
        )
        drainer.drain(n=self.n * 10)

        # Get Data and assert result is as expected
        model_fg, I = self.ap3.build_graph_pattern_final(), drainer.firing_graph.Iw
        track_ib = drainer.firing_graph.backward_firing['i']

        # Make sure that drained structure is as expected
        self.assertTrue((drainer.firing_graph.I.toarray() == model_fg.I.toarray()).all())

        # Test for forward and backward firing tracking and coherence
        self.assertTrue(
            all([I[j, 1] == track_ib[j, 1] + self.w0 for j in self.ap3.target[0]
                 if j not in self.ap3.target_selected[0]])
        )
        self.assertTrue(
            all([I[j, 4] == track_ib[j, 4] + self.w0 for j in self.ap3.target[1]
                 if j not in self.ap3.target_selected[1]])
        )

        # VISUAL TEST:
        if self.visual:
            # GOT
            fg_got = self.ap3.build_graph_pattern_final()
            ax_graph_got = gather_matrices(fg_got.Iw.toarray(), fg_got.Dw.toarray(), fg_got.Ow.toarray())
            plot_graph(ax_graph_got, self.ap3.layout(), title='GOT')

            # Fring Graph at convergence
            ax_graph_conv = gather_matrices(
                self.ap3_fg.Iw.toarray(), self.ap3_fg.Dw.toarray(), self.ap3_fg.Ow.toarray()
            )

            plot_graph(ax_graph_conv, self.ap3.layout(), title='Result Test')

    def test_batch_size(self):
        """
        Test batch size coherence (depth 3)
        python -m unittest tests.unit.core.test_drainer.TestDrainer.test_batch_size

        """
        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.ap3.generate_io_sequence(1000, seed=1234)
        server = create_server(csc_matrix(ax_input), csc_matrix(ax_output))

        # Drain with batch size of 2
        drainer_2 = FiringGraphDrainer(
            self.ap3_fg.copy(), server, t=1000, p=np.ones(self.ap3.no), r=np.ones(self.ap3.no), batch_size=2
        )
        drainer_2.drain(n=100)

        # Drain with batch size of 1
        server.stream_features()
        drainer_1 = FiringGraphDrainer(
            self.ap3_fg.copy(), server, t=1000, p=np.ones(self.ap3.no), r=np.ones(self.ap3.no), batch_size=1
        )
        drainer_1.drain(n=200)

        # There should be no more difference between edges weight than difference of batch size
        ax_diff = (drainer_1.firing_graph.Iw.toarray() - drainer_2.firing_graph.Iw.toarray())
        self.assertTrue(((-2 < ax_diff) & (ax_diff < 2)).all())

        # Drain with manual iteration management
        server.stream_features()
        drainer_1m = FiringGraphDrainer(
            self.ap3_fg.copy(), server, t=1000, p=np.ones(self.ap3.no), r=np.ones(self.ap3.no), batch_size=1
        )
        for _ in range(200):
            drainer_1m.drain()
            drainer_1m.reset_all()

        # Compare to auto iteration, no difference in edges weight higher than the one due to inertia of auto iter*
        ax_diff = (drainer_1.firing_graph.Iw.toarray() - drainer_1m.firing_graph.Iw.toarray())
        self.assertTrue(((-5 < ax_diff) & (ax_diff < 5)).all())

    def test_multi_output(self):
        """
        Test setting different p and r for different
        python -m unittest tests.unit.core.test_drainer.TestDrainer.test_multi_output

        """
        # Create I/O and save it into tmpdir files
        server = create_server(self.sax_forward, self.sax_backward)

        # Create drainer
        drainer = FiringGraphDrainer(self.fg, server, p=self.p, r=self.r, t=4*10, batch_size=4*10).drain_all(n_max=4*10)

        for i in range(2):
            self.assertEqual(drainer.firing_graph.Iw[i, i], self.weight + 10 * self.r[i])

        for i in range(2, 4):
            self.assertEqual(drainer.firing_graph.Iw[i, i], self.weight - 10 * self.p[i])


def create_server(sax_in, sax_out):

    # Create and init servers
    server = ArrayServer(sax_in, sax_out)
    server.stream_features()

    return server

