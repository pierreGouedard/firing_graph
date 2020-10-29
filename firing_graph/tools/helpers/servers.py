# Global imports
import pickle
from scipy.sparse import vstack, csc_matrix
from numpy import int8, uint8, vectorize
from numpy.random import binomial

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
                 pattern_backward=None, sax_mask=None, dropout_rate_mask=0):

        # Set sparse signals
        self.n_label = sax_backward.shape[1]
        self.__sax_forward = sax_forward.tocsr()
        self.__sax_backward = sax_backward
        self.__sax_mask = sax_mask
        self.dtype_forward = dtype_forward
        self.dtype_backward = dtype_backward

        # Initialize data
        self.sax_data_forward = None
        self.sax_data_backward = None
        self.sax_mask_forward = None

        # Set preprocessing patterns
        self.pattern_forward, self.pattern_backward = pattern_forward, pattern_backward
        self.dropout_rate_mask = dropout_rate_mask

        # Define streaming features
        self.step_forward, self.step_backward = 0, 0

    def update_mask(self, pattern_mask):
        if pattern_mask is None:
            return

        sax_mask = pattern_mask.propagate(self.__sax_forward, return_activations=False)
        if self.__sax_mask is not None:
            self.__sax_mask += sax_mask.astype(uint8)
        else:
            self.__sax_mask = sax_mask.astype(uint8)

    def stream_features(self):
        self.step_forward, self.step_backward = 0, 0
        return self

    def check_synchro(self):
        assert self.step_forward == self.step_backward, "[SERVER]: Step of forward and backward differs"
        return self

    def synchonize_steps(self):
        step = max(self.step_forward, self.step_backward)
        self.step_forward, self.step_backward = step, step

    @staticmethod
    def recursive_positions(step, n, n_max):

        l_positions = [(step, min(step + n, n_max))]
        residual = max(step + n - n_max, 0)

        if residual > 0:
            l_positions.extend(ArrayServer.recursive_positions(0, residual, n_max))

        return l_positions

    def next_forward(self, n=1, update_step=True):

        # Compute indices
        l_positions = self.recursive_positions(self.step_forward, n, self.__sax_forward.shape[0])
        sax_data = vstack([self.__sax_forward[start:end, :].tocsr() for (start, end) in l_positions])

        # Compute new step of forward
        if update_step:
            self.step_forward = (self.step_forward + n) % self.__sax_forward.shape[0]

        # Get process forward data
        if self.pattern_forward is not None:
            sax_data = self.pattern_forward.propagate(sax_data)

        self.sax_data_forward = sax_data.astype(self.dtype_forward).tocsr()

        return self

    def next_backward(self, n=1, update_step=True):
        # Compute indices
        l_positions = self.recursive_positions(self.step_backward, n, self.__sax_backward.shape[0])
        sax_data = vstack([self.__sax_backward[start:end, :].tocsr() for (start, end) in l_positions]).astype(int8)

        # Get process backward data
        if self.pattern_backward is not None:
            sax_data = self.pattern_backward.propagate(sax_data)

        self.sax_data_backward = sax_data.astype(self.dtype_backward)

        # Get mask data
        if self.__sax_mask is not None:
            sax_mask = vstack([self.__sax_mask[start:end, :].tocsr() for (start, end) in l_positions])
            if self.dropout_rate_mask > 0:
                dropout_func = vectorize(lambda x: binomial(int(x), 1 - self.dropout_rate_mask) > 0 if x > 0 else 0)
                sax_mask.data = dropout_func(sax_mask.data)
                sax_mask.eliminate_zeros()

            if self.pattern_backward is not None:
                sax_mask = self.pattern_backward.propagate(sax_mask.astype(self.dtype_forward))

            self.sax_mask_forward = sax_mask.astype(self.dtype_backward)

        # Compute new step of backward
        if update_step:
            self.step_backward = (self.step_backward + n) % self.__sax_backward.shape[0]

        return self

    def save_as_pickle(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load_pickle(path):
        with open(path, 'rb') as handle:
            server = pickle.load(handle)

        return server
