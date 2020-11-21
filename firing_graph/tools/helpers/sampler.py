# Global import
import numpy as np
import time
from scipy.sparse import hstack, csc_matrix

# local import

class Sampler(object):
    """
        This class implements a supervised sampler engine. It samples randomly input bit base on a concomitant activation
        of bit and output bit.
    """

    def __init__(self, n_label, n_inputs, p_sample=0.8, verbose=0):
        # Sampling parameters
        self.n_label, self.n_inputs, self.p_sample = n_label, n_inputs, p_sample

        # Utils parameters
        self.verbose = verbose

    def sample(self):
        raise NotImplementedError