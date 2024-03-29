# Global import
from scipy.sparse import csr_matrix
import numpy as np

# Local import
from .linalg.forward import fti, fpo
from .linalg.backward import bui


class FiringGraphDrainer(object):
    def __init__(self, firing_graph, server, batch_size, penalties=None, rewards=None, verbose=0):

        # Core params
        self.ax_p, self.ax_r = penalties, rewards
        self.bs = batch_size
        self.firing_graph = firing_graph
        self.verbose = verbose

        # stream feed forward and backward
        self.server = server
        self.server.stream_features()

        # Init signals
        self.sax_i, self.sax_c, self.sax_cb = init_forward_signal(self.firing_graph, self.bs)
        self.iter = 0

    def reset(self, server=False):
        self.sax_i, self.sax_c, self.sax_cb = init_forward_signal(self.firing_graph, self.bs)
        self.iter = 0
        if server:
            self.server.stream_features()

    def drain(self, n=1):
        early_stopping, j = False, 0
        while j < n:
            self.run_iteration()

            # Condition of stop that has to be put in the drainer
            if self.firing_graph.I.nnz == 0:
                early_stopping = True
                self.server.synchonize_steps()
                break

            # Increment count iteration
            j += 1

        # Make sure forward and backward signal are synchronized.
        if not early_stopping:
            self.server.check_synchro()

        return self

    def run_iteration(self):
        # Forward pass
        self.forward()

        # Backward pass
        self.backward()

        # Increment iteration nb
        self.iter += 1

    def forward(self):
        # Get new input
        self.sax_i = self.server.next_forward(n=self.bs).sax_data_forward

        # transmit to core vertices
        self.sax_c = fti(self.firing_graph.I, self.sax_i, self.firing_graph.levels)

        # Compute feedback
        self.sax_cb = fpo(self.sax_c, self.server.next_backward(n=self.bs).sax_data_backward, self.ax_p, self.ax_r)

    def backward(self):
        self.firing_graph.update_backward_count(bui(self.sax_cb, self.sax_i, self.firing_graph))
        self.firing_graph.refresh_matrices()


def init_forward_signal(fg, batch_size):
    if fg is not None:
        return (
                   csr_matrix((batch_size, fg.I.shape[0]), dtype=bool),
                   csr_matrix((batch_size, fg.C.shape[0]), dtype=bool),
                   csr_matrix((batch_size, fg.C.shape[0]), dtype=np.int32)
        )
    return None, None, None
