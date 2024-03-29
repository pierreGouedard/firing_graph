# Global imports
import pickle
from numpy import arange, hstack, where
from random import choice

# Local imports


class ArrayServer(object):
    def __init__(
            self, sax_forward, sax_backward, pattern_forward=None, pattern_backward=None, dropout_rate_mask=0,
            mask_method="count"
    ):

        # Set meta
        self.n_label = sax_backward.shape[1]
        self.n_inputs = sax_forward.shape[1]

        # Set sparse signals
        self.sax_forward = sax_forward.tocsr()
        self.sax_backward = sax_backward.tocsr()

        self.sax_pattern_forward = None
        if pattern_forward is not None:
            self.sax_pattern_forward = pattern_forward.seq_propagate(self.sax_forward)

        self.sax_pattern_backward = None
        if pattern_backward is not None:
            self.sax_pattern_backward = pattern_backward.seq_propagate(self.sax_backward)

        # Set mask parameters
        self.mask_method = mask_method
        self.ax_param_mask = None
        self.__ax_mask = None

        # Initialize data
        self.sax_data_forward = None
        self.sax_data_backward = None

        # Set preprocessing patterns
        self.pattern_forward, self.pattern_backward = pattern_forward, pattern_backward
        self.dropout_rate_mask = dropout_rate_mask

        # Define streaming features
        self.step_forward, self.step_backward = 0, 0

    def update_pattern_backward(self, pattern_backward):
        self.pattern_backward = pattern_backward
        self.sax_pattern_backward = pattern_backward.seq_propagate(self.sax_backward)

    def get_random_samples(self, n):
        if self.__ax_mask is not None:
            l_pool = list(self.__ax_mask.nonzero()[0])

        else:
            l_pool = list(set(range(self.sax_forward.shape[0])))

        # For test only
        #l_pool = self.sax_backward[:, 1].nonzero()[0]
        
        return self.get_sub_forward([choice(l_pool) for _ in range(n)])

    def get_sub_forward(self, indices):
        return self.sax_forward[indices, :]

    def count_unmasked(self,):
        if self.__ax_mask is None:
            return self.sax_forward.shape[0]
        else:
            return self.__ax_mask.sum()

    def update_mask(self):
        assert self.step_backward == self.step_forward, "Can't update mask if forward and backward not sync"
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

    def next_forward(self, n=1, update_step=True):

        # get indices
        ax_indices = self.recursive_positions(self.step_forward, n, self.sax_forward.shape[0])

        if self.sax_pattern_forward is not None:
            self.sax_data_forward = self.sax_pattern_forward[ax_indices, :]
        else:
            self.sax_data_forward = self.sax_forward[ax_indices, :]

        # Compute new step of forward
        if update_step:
            self.step_forward = (ax_indices[-1] + 1) % self.sax_forward.shape[0]

        return self

    def next_all_forward(self):
        return self.next_forward(self.count_unmasked(), update_step=False)

    def next_all_backward(self):
        return self.next_backward(self.count_unmasked(), update_step=False)

    def next_backward(self, n=1, update_step=True):

        # get indices
        ax_indices = self.recursive_positions(self.step_backward, n, self.sax_forward.shape[0])

        if self.sax_pattern_backward is not None:
            self.sax_data_backward = self.sax_pattern_backward[ax_indices, :]
        else:
            self.sax_data_backward = self.sax_backward[ax_indices, :]

        # Compute new step of backward
        if update_step:
            self.step_backward = (ax_indices[-1] + 1) % self.sax_forward.shape[0]

        return self

    def save_as_pickle(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load_pickle(path):
        with open(path, 'rb') as handle:
            server = pickle.load(handle)

        return server
