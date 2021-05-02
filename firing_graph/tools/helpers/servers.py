# Global imports
import pickle
from scipy.sparse import vstack
from numpy import int8, uint16, vectorize, arange, hstack, where
from numpy.random import binomial
from random import choice

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
                 pattern_backward=None, dropout_rate_mask=0, mask_method="count"):

        # Set meta
        self.n_label = sax_backward.shape[1]
        self.n_inputs = sax_forward.shape[1]

        # Set sparse signals
        self.__sax_forward = sax_forward.tocsr()
        self.__sax_backward = sax_backward
        self.dtype_forward = dtype_forward
        self.dtype_backward = dtype_backward

        # Set mask parameters
        self.mask_method = mask_method
        self.ax_param_mask = None
        self.__ax_mask = None

        # Initialize data
        self.sax_data_forward = None
        self.sax_data_backward = None
        self.sax_mask_forward = None

        # Set preprocessing patterns
        self.pattern_forward, self.pattern_backward = pattern_forward, pattern_backward
        self.dropout_rate_mask = dropout_rate_mask

        # Define streaming features
        self.step_forward, self.step_backward = 0, 0

    def propagate_all(self, pattern, ax_values=None, return_activations=False):
        if ax_values is not None:
            sax_res = pattern.propagate_values(self.__sax_forward, ax_values=ax_values)
        else:
            sax_res = pattern.propagate(self.__sax_forward, return_activations=return_activations)

        return sax_res

    def get_random_samples(self, n):
        if self.__ax_mask is not None:
            l_pool = list(self.__ax_mask.nonzero()[0])

        else:
            l_pool = list(set(range(self.__sax_forward.shape[0])))

        return self.__sax_forward[[choice(l_pool) for _ in range(n)], :]

    def count_unmasked(self,):
        if self.__ax_mask is None:
            return self.__ax_mask.shape[0]
        else:
            return self.__ax_mask.sum()

    def update_mask(self):
        self.__ax_mask = self.apply_mask_method(self.ax_param_mask.copy())

    def apply_mask_method(self, ax_mask):
        pass

    def stream_features(self):
        self.step_forward, self.step_backward = 0, 0
        return self

    def check_synchro(self):
        assert self.step_forward == self.step_backward, "[SERVER]: Step of forward and backward differs"
        return self

    def synchonize_steps(self):
        step = max(self.step_forward, self.step_backward)
        self.step_forward, self.step_backward = step, step

    def recursive_positions(self, step, n, n_max):

        # Apply mask to data
        if self.__ax_mask is not None:
            ax_cum_mask = self.__ax_mask[step:].cumsum()
            end = step + where(ax_cum_mask == min(n, ax_cum_mask.max()))[0].max()
            ax_positions = arange(step, end + 1)[self.__ax_mask[step:end + 1]]

        else:
            ax_positions = arange(step, min(step + n, n_max))

        residual = max(n - ax_positions.shape[0], 0)

        if residual > 0:
            ax_positions = hstack([ax_positions, self.recursive_positions(0, residual, n_max)])

        return ax_positions

    def get_init_precision(self):
        sax_masked_backward = self.__sax_backward.astype(int)
        if self.__ax_mask is not None:
            sax_masked_backward = sax_masked_backward[self.__ax_mask, :].astype(int)

        return (sax_masked_backward > 0).sum(axis=0).A[0] / sax_masked_backward.shape[0]

    def next_forward(self, n=1, update_step=True):

        # get indices
        ax_indices = self.recursive_positions(self.step_forward, n, self.__sax_forward.shape[0])
        sax_data = self.__sax_forward[ax_indices, :]

        # Propagate in pattern if any
        if self.pattern_forward is not None:
            sax_data = self.pattern_forward.propagate(sax_data)

        # Set data type
        self.sax_data_forward = sax_data.astype(self.dtype_forward).tocsr()

        # Compute new step of forward
        if update_step:
            self.step_forward = (ax_indices[-1] + 1) % self.__sax_forward.shape[0]

        return self

    def next_backward(self, n=1, update_step=True):

        # get indices
        ax_indices = self.recursive_positions(self.step_backward, n, self.__sax_forward.shape[0])
        sax_data = self.__sax_backward[ax_indices, :]

        # Propagate in pattern if any
        if self.pattern_backward is not None:
            sax_data = self.pattern_backward.propagate(sax_data)

        # Set data type
        self.sax_data_backward = sax_data.astype(self.dtype_backward).tocsc()

        # Compute new step of backward
        if update_step:
            self.step_backward = (ax_indices[-1] + 1) % self.__sax_forward.shape[0]

        return self

    def save_as_pickle(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load_pickle(path):
        with open(path, 'rb') as handle:
            server = pickle.load(handle)

        return server
