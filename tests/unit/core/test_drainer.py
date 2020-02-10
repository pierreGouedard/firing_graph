# Global imports
import unittest
from scipy.sparse import csc_matrix

# Local import
from core.data_structure.utils import gather_matrices
from core.tools.helpers.servers import ArrayServer
from core.solver.drainer import FiringGraphDrainer
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

    def test_time_mask(self):

        """
        Test the well functioning of mask on backward updates
        python -m unittest tests.unit.core.test_drainer.TestDrainer.test_time_mask

        """

        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.ap2.generate_io_sequence(1000, seed=1234)
        server = create_server(csc_matrix(ax_input), csc_matrix(ax_output))

        # Create drainer
        drainer = FiringGraphDrainer(self.ap2_fg, server, t=self.t_mask, p=1, q=1, batch_size=1, verbose=1)
        drainer.drain(100)

        # Get matrice of the graph
        fg, fg_final, fg_init = drainer.firing_graph, self.ap2.build_graph_pattern_final(), self.ap2.build_graph_pattern_init()
        I, I_init, I_final = fg.Iw.toarray(), fg_init.Iw.toarray(), fg_final.Iw.toarray()

        # Assert mask are working (no more than self.t_mask structure update
        self.assertTrue((I[I >= self.w0] == I_final[I_final > 0] + self.t_mask).all())
        self.assertTrue((I[(0 < I) & (I <= self.w0)] == I_init[~(I_final > 0) & (I_init > 0)] - self.t_mask).all())

    def test_andpattern2(self):

        """
        Test And Pattern of depth 2
        python -m unittest tests.unit.core.test_drainer.TestDrainer.test_andpattern2

        """

        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.ap2.generate_io_sequence(1000, seed=1234)
        server = create_server(csc_matrix(ax_input), csc_matrix(ax_output))

        # Create drainer
        drainer = FiringGraphDrainer(self.ap2_fg, server, t=1000, p=1, q=1, batch_size=10, verbose=1)
        drainer.drain(n=self.n)

        # Get Data and assert result is as expected
        model_fg, I = self.ap2.build_graph_pattern_final(), drainer.firing_graph.Iw
        track_ib = drainer.firing_graph.backward_firing['i']

        # Check correctness of structure
        self.assertTrue((drainer.firing_graph.I.toarray() == model_fg.I.toarray()).all())

        # Test firing tracker
        self.assertTrue(all([I[j, 0] == track_ib[j, 0] + self.w0 for j in self.ap2.target[0]]))
        self.assertTrue(all([I[j, 1] == track_ib[j, 1] + self.w0 for j in self.ap2.target[1]]))

        # VISUAL TEST:
        if self.visual:
            # GOT
            fg_got = self.ap2.build_graph_pattern_final()
            ax_graph_got = gather_matrices(fg_got.Iw.toarray(), fg_got.Dw.toarray(), fg_got.Ow.toarray())
            plot_graph(ax_graph_got, self.ap2.layout(), title='GOT')

            # Fring Graph at convergence
            ax_graph_conv = gather_matrices(
                self.ap2_fg.Iw.toarray(), self.ap2_fg.Dw.toarray(), self.ap2_fg.Ow.toarray()
            )

            plot_graph(ax_graph_conv, self.ap2.layout(), title='Result Test')

    def test_andpattern3(self):
        """
        Test And Pattern of depth 3
        python -m unittest tests.unit.core.test_drainer.TestDrainer.test_andpattern3

        """
        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.ap3.generate_io_sequence(1000, seed=1234)
        server = create_server(csc_matrix(ax_input), csc_matrix(ax_output))

        # Create drainer
        drainer = FiringGraphDrainer(self.ap3_fg, server, t=1000, p=1, q=1, batch_size=1, verbose=1)
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

    def test_batch_size_2(self):
        """
        Test batch size coherence (depth 2)
        python -m unittest tests.unit.core.test_drainer.TestDrainer.test_batch_size_2

        """
        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.ap2.generate_io_sequence(1000, seed=1234)
        server = create_server(csc_matrix(ax_input), csc_matrix(ax_output))

        # Drain with batch size of 2
        drainer_2 = FiringGraphDrainer(self.ap2_fg.copy(), server, t=1000, p=1, q=1, batch_size=2, verbose=1)
        drainer_2.drain(n=200)

        # Drain with batch size of 1
        server.stream_features()
        drainer_1 = FiringGraphDrainer(self.ap2_fg.copy(), server, t=1000, p=1, q=1, batch_size=1, verbose=1)
        drainer_1.drain(n=400)

        # There should be no more difference between edges weight than difference of batch size
        ax_diff = (drainer_1.firing_graph.Iw.toarray() - drainer_2.firing_graph.Iw.toarray())
        self.assertTrue(((-2 < ax_diff) & (ax_diff < 2)).all())

        # Drain with manual iteration management
        server.stream_features()
        drainer_1m = FiringGraphDrainer(self.ap2_fg.copy(), server, t=1000, p=1, q=1, batch_size=1, verbose=1)
        for _ in range(400):
            drainer_1m.drain()
            drainer_1m.reset_all()

        # Compare to auto iteration, no difference in edges weight higher than the one due to inertia of auto iter*
        ax_diff = (drainer_1.firing_graph.Iw.toarray() - drainer_1m.firing_graph.Iw.toarray())
        self.assertTrue(((-3 < ax_diff) & (ax_diff < 3)).all())

    def test_batch_size_3(self):
        """
        Test batch size coherence (depth 3)
        python -m unittest tests.unit.core.test_drainer.TestDrainer.test_batch_size_3

        """
        # Create I/O and save it into tmpdir files
        ax_input, ax_output = self.ap3.generate_io_sequence(1000, seed=1234)
        server = create_server(csc_matrix(ax_input), csc_matrix(ax_output))

        # Drain with batch size of 2
        drainer_2 = FiringGraphDrainer(self.ap3_fg.copy(), server, t=1000, p=1, q=1, batch_size=2, verbose=1)
        drainer_2.drain(n=100)

        # Drain with batch size of 1
        server.stream_features()
        drainer_1 = FiringGraphDrainer(self.ap3_fg.copy(), server, t=1000, p=1, q=1, batch_size=1, verbose=1)
        drainer_1.drain(n=200)

        # There should be no more difference between edges weight than difference of batch size
        ax_diff = (drainer_1.firing_graph.Iw.toarray() - drainer_2.firing_graph.Iw.toarray())
        self.assertTrue(((-2 < ax_diff) & (ax_diff < 2)).all())

        # Drain with manual iteration management
        server.stream_features()
        drainer_1m = FiringGraphDrainer(self.ap3_fg.copy(), server, t=1000, p=1, q=1, batch_size=1, verbose=1)
        for _ in range(200):
            drainer_1m.drain()
            drainer_1m.reset_all()

        # Compare to auto iteration, no difference in edges weight higher than the one due to inertia of auto iter*
        ax_diff = (drainer_1.firing_graph.Iw.toarray() - drainer_1m.firing_graph.Iw.toarray())
        self.assertTrue(((-5 < ax_diff) & (ax_diff < 5)).all())


# *The inertia cited here refer to the fact that for automatic iteration, forward signal are sent subsequently. Thus a
# signal can be sent even if it go through an edge that it is supposed to be removed from feedback of signal sent before
# Using a manual iteration, we wait for the feedback of each forward signal before sending new forward signal.

def create_server(sax_in, sax_out):

    # Create and init servers
    server = ArrayServer(sax_in, sax_out)
    server.stream_features()

    return server

