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
        self.sax_test = csc_matrix([[1], [0], [1], [0], [1], [1], [1], [0]])

        # Create patterns to test functionality of forward and  backward patterns in server
        d_matrices = create_empty_matrices(self.sax_forward.shape[1], self.sax_backward.shape[1], 1)
        d_matrices['Iw'][3, 0], d_matrices['Ow'][0, 0] = 1, 1
        self.pattern = FiringGraph('test_server', np.ones(1), d_matrices, depth=2)

        # Create server
        self.server = ArrayServer(
            self.sax_forward, self.sax_backward, dtype_forward=int, dtype_backward=int,
            pattern_mask=self.pattern.copy()
        )

        self.sax_expected = np.array([-1, 0, -1, 0, 1, 1, 1, 0])
        self.sax_expected_mask = np.array([-1, 0, 0, 0, 1, 1, 0, 0])

    def test_server_no_mask(self):
        """
        python -m unittest tests.unit.core.test_server.TestServer.test_server_no_mask

        """
        # stream once
        self.server.pattern_mask = None
        sax_i = self.server.next_forward(5).sax_data_forward

        # Assert correct behaviour
        self.assertEqual(sax_i.shape, (5, self.sax_forward.shape[1]))
        self.assertTrue((sax_i.toarray() == self.sax_forward.toarray()[:5, :]).all())
        self.assertEqual(self.server.step_forward, 5)

        # Stream another time
        sax_i = self.server.next_forward(5).sax_data_forward

        # Assert correct behaviour
        self.assertEqual(sax_i.shape, (5, self.sax_forward.shape[1]))
        self.assertTrue((sax_i.toarray()[0, :] == self.sax_forward.toarray()[5, :]).all())
        self.assertTrue((sax_i.toarray()[-1, :] == self.sax_forward.toarray()[9 % self.sax_forward.shape[0], :]).all())
        self.assertEqual(self.server.step_forward, 2)

        # Stream feedback signal
        self.server.stream_features()

        # Assert correct behaviour
        sax_ob = fpo(self.sax_test, self.server, self.sax_forward.shape[0], np.ones(1), np.ones(1))
        self.assertTrue((sax_ob.toarray()[0] == self.sax_expected).all())

    def test_server_mask_pattern(self):
        """
        python -m unittest tests.unit.core.test_server.TestServer.test_server_mask_pattern

        """

        # Stream feedback signal with soft orthogonality settings
        self.server.stream_features()

        # Assert correct behaviour
        sax_ob = fpo(self.sax_test, self.server, self.sax_forward.shape[0], np.ones(1), np.ones(1))
        self.assertTrue((sax_ob.toarray() == self.sax_expected_mask).all())
