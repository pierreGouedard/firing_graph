# Global imports
import unittest
import numpy as np
from scipy.sparse import csc_matrix

# Local import
from core.tools.helpers.servers import ArrayServer
from core.data_structure.graph import FiringGraph
from core.data_structure.utils import create_empty_matrices
from core.tools.equations.forward import fpo


class TestServer(unittest.TestCase):
    def setUp(self):

        # Create input and output signal
        self.sax_forward = csc_matrix([
            [1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1],
            [0, 1, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 1],
            [1, 1, 0, 1, 1, 1],

        ])

        self.sax_backward = csc_matrix([[0], [0], [0], [0], [1], [1], [1], [1]])
        self.sax_test = csc_matrix([[1], [0], [1], [0], [1], [1], [1], [1]])

        # Create patterns to test functionality of forward and  backward patterns in server
        d_matrices = create_empty_matrices(self.sax_forward.shape[1], self.sax_backward.shape[1], 1)
        d_matrices['Iw'][3, 0], d_matrices['Ow'][0, 0] = 1, 1
        self.pattern = FiringGraph('test_server', np.ones(1), d_matrices, depth=2)

        self.soft_server = ArrayServer(
            self.sax_forward, self.sax_backward, dtype_forward=int, dtype_backward=int,
            pattern_backward=self.pattern.copy(), strat_colinearity='soft'
        )

        self.hard_server = ArrayServer(
            self.sax_forward, self.sax_backward, dtype_forward=int, dtype_backward=int,
            pattern_backward=self.pattern.copy(), strat_colinearity='hard'
        )

        self.sax_nopattern_expected = np.array([-1, 0, -1, 0, 1, 1, 1, 1])
        self.sax_soft_expected = np.array([-1, 0, -1, 0, 1, 1, 0, 0])
        self.sax_hard_expected = np.array([-1, 0, -1, 0, 1, 1, -1, -1])

    def test_server_no_pattern(self):
        """
        python -m unittest tests.unit.core.test_server.TestServer.test_server_no_pattern

        """
        # stream once
        self.soft_server.pattern_backward = None
        sax_i = self.soft_server.next_forward(5)

        # Assert correct behaviour
        self.assertEqual(sax_i.shape, (5, self.sax_forward.shape[1]))
        self.assertTrue((sax_i.toarray() == self.sax_forward.toarray()[:5, :]).all())
        self.assertEqual(self.soft_server.step_forward, 5)

        # Stream another time
        sax_i = self.soft_server.next_forward(5)

        # Assert correct behaviour
        self.assertEqual(sax_i.shape, (5, self.sax_forward.shape[1]))
        self.assertTrue((sax_i.toarray()[0, :] == self.sax_forward.toarray()[5, :]).all())
        self.assertTrue((sax_i.toarray()[-1, :] == self.sax_forward.toarray()[9 % self.sax_forward.shape[0], :]).all())

        # Stream feedback signal
        self.soft_server.stream_features()
        self.soft_server.pattern_backward = None

        # Assert correct behaviour
        sax_ob = fpo(self.sax_test, self.soft_server, self.sax_forward.shape[0], 1, 1)
        self.assertTrue((sax_ob.toarray()[0] == self.sax_nopattern_expected).all())

    def test_server_backward_pattern(self):
        """
        python -m unittest tests.unit.core.test_server.TestServer.test_server_backward_pattern

        """

        # Stream feedback signal with soft orthogonality settings
        self.soft_server.stream_features()

        # Assert correct behaviour
        sax_ob = fpo(self.sax_test, self.soft_server, self.sax_forward.shape[0], 1, 1)
        self.assertTrue((sax_ob.toarray() == self.sax_soft_expected).all())

        # Stream feedback signal with soft orthogonality settings
        self.hard_server.stream_features()

        # Assert correct behaviour
        sax_ob = fpo(self.sax_test, self.hard_server, self.sax_forward.shape[0], 1, 1)
        self.assertTrue((sax_ob.toarray() == self.sax_hard_expected).all())


