# Global imports
import pickle
from scipy.sparse import csr_matrix, vstack

# Local imports


class FileServer(object):

    def __init__(self, path_forward, path_backward, streamer, is_cyclic=True):

        # Define core attribute of the file server
        self.path_forward = path_forward
        self.path_backward = path_backward
        self.streamer = streamer
        self.is_cyclic = is_cyclic

        # Initialize stream to None
        self.stream_forward, self.stream_backward = None, None

    def stream_data(self):

        # Stream I/O
        self.stream_forward = self.streamer.create_stream(
            self.path_forward, is_cyclic=self.is_cyclic, orient='row'
        )

        self.stream_backward = self.streamer.create_stream(
            self.path_backward, is_cyclic=self.is_cyclic, orient='row'
        )

    def next_forward(self):
        return self.stream_forward.next()

    def next_backward(self):
        return self.stream_backward.next()

    def save_as_pickle(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load_pickle(path):

        with open(path, 'rb') as handle:
            server = pickle.load(handle)

        return server


class ArrayServer(object):

    def __init__(self, sax_forward, sax_backward, dtype_forward=int, dtype_backward=int, pattern_forward=None,
                 pattern_backward=None, strat_colinearity='soft'):

        # Set sparse signals
        self.sax_forward = sax_forward.tocsr()
        self.sax_backward = sax_backward
        self.dtype_forward = dtype_forward
        self.dtype_backward = dtype_backward

        # Set preprocessing patterns
        self.pattern_forward, self.pattern_backward = pattern_forward, pattern_backward

        # Define streaming features
        self.step_forward, self.step_backward = 0, 0

        # Define strategy to penalize co-linearity of output with signal propagated by pattern_backward
        self.strat_colinearity = strat_colinearity

    def stream_features(self):
        self.step_forward, self.step_backward = 0, 0

    def check_synchro(self):
        assert self.step_forward == self.step_backward, "[SERVER]: Step of forward and backward differs"

    @staticmethod
    def recursive_positions(step, n, n_max):

        l_positions = [(step, min(step + n, n_max))]
        residual = max(step + n - n_max, 0)

        if residual > 0:
            l_positions.extend(ArrayServer.recursive_positions(0, residual, n_max))

        return l_positions

    def next_forward(self, n=1):

        # Compute indices
        l_positions = self.recursive_positions(self.step_forward, n, self.sax_forward.shape[0])
        sax_data = vstack([self.sax_forward[start:end, :] for (start, end) in l_positions])

        # Compute new step of forward
        self.step_forward = (self.step_forward + n) % self.sax_forward.shape[0]

        if self.pattern_forward is not None:
            sax_data = self.pattern_forward.propagate(sax_data)

        return sax_data.astype(self.dtype_forward).tocsr()

    def next_backward(self, sax_o=None,  n=1):

        # Compute indices
        l_positions = self.recursive_positions(self.step_backward, n, self.sax_backward.shape[0])
        sax_data = vstack([self.sax_backward[start:end, :] for (start, end) in l_positions])

        # Compute new step of backward
        self.step_backward = (self.step_backward + n) % self.sax_backward.shape[0]

        if sax_o is None:
            return sax_data.astype(self.dtype_backward)

        # Compute post processing signal from backward pattern
        if self.pattern_backward is not None:
            sax_pattern = vstack([self.sax_forward[start:end, :] for (start, end) in l_positions])
            sax_pattern = self.pattern_backward.propagate(sax_pattern).multiply(sax_data)

            if self.strat_colinearity == 'soft':
                return sax_data.astype(self.dtype_backward), (sax_o - sax_pattern > 0).astype(self.dtype_backward)

            else:
                return (sax_data - sax_pattern > 0).astype(self.dtype_backward), (sax_o > 0).astype(self.dtype_backward)

        else:
            return sax_data.astype(self.dtype_backward), (sax_o > 0).astype(self.dtype_backward)

    def save_as_pickle(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load_pickle(path):

        with open(path, 'rb') as handle:
            server = pickle.load(handle)

        return server






