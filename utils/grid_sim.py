# Global imports
from scipy.sparse import csc_matrix
import numpy as np

# Local import
from core.tools.helpers.servers import ArrayServer
from core.tools.helpers.drivers import FileDriver


class GridSim(object):
    driver = FileDriver('grid_sim_driver', 'File driver for simulation')

    def __init__(self, name):
        self.name = name
        self.server = None

    def phi(self, omega):
        raise NotImplementedError

    def omega(self, i):
        raise NotImplementedError

    def nu(self, i):
        raise NotImplementedError

    def mu(self, i):
        raise NotImplementedError

    def N(self, t, i):
        raise NotImplementedError

    def generate_io_sequence(self, n, mask_target=None):
        raise NotImplementedError

    def stream_io_sequence(self, n, mask_target=None):
        sax_in, sax_out = self.generate_io_sequence(n, mask_target=mask_target)
        return self.create_server(sax_in, sax_out)

    def create_server(self, sax_in, sax_out):

        # Create and init server
        server = ArrayServer(sax_in, sax_out)
        server.stream_features()

        return server


class SignalPlusNoiseGrid(GridSim):

    def __init__(self, n_sim, n_bits, p_target, n_targets, p_noise):

        # Size of the simulation
        self.n_sim, self.n_bits, self.n_targets = n_sim, n_bits, n_targets

        # Base param of simulation
        self.p_target, self.p_noise = p_target, p_noise
        self.target_bits = [np.random.choice(range(self.n_bits), n_targets, replace=False) for _ in range(self.n_sim)]

        # Set base values of p and q, used in score process
        self.p, self.q = 1, 1

        GridSim.__init__(self, 'SignalPlusNoiseGrid')

    def phi(self, omega):
        return self.p_target / (self.p_target + (1 - self.p_target) * omega)

    def mu(self, j):
        return pow(self.p_noise, j)

    def mean_score_signal(self, t, i):
        return self.N(t, i) + int(t * (self.phi(self.omega(i) - self.delta(i)) * (self.p + self.q) - self.p))

    def mean_score_noise(self, t, i):
        return self.N(t, i) + int(t * (self.phi(self.omega(i) + self.delta(i)) * (self.p + self.q) - self.p))

    def omega(self, i):
        return (1 + self.p_noise) * pow(self.p_noise, i) / 2.

    def delta(self, i):
        return (1 - self.p_noise) * pow(self.p_noise, i) / 2

    def N(self, t, i):
        return - int(t * (self.phi(self.omega(i)) * (self.p + self.q) - self.p))

    def set_score_params(self, i):
        for q in range(1000):
            p = np.ceil(q * self.phi(self.omega(i)) / (1 - self.phi(self.omega(i))))

            score = (self.phi(self.omega(i) - self.delta(i)) * (p + q)) - p

            if score > 0.:
                self.p, self.q = p, q
                break

        return self

    def generate_io_sequence(self, n, mask_target=None):

        ax_inoise, ax_onoise = self.generate_noise_sequence(n)
        ax_itarget, ax_otarget = self.generate_target_sequence(n)

        ax_inputs = (ax_inoise + ax_itarget) > 0
        ax_outputs = (ax_onoise + ax_otarget) > 0

        return csc_matrix(ax_inputs, dtype=bool), csc_matrix(ax_outputs, dtype=bool)

    def generate_noise_sequence(self, n):
        # Init noisy sequence
        ax_outputs = np.zeros((n, self.n_sim))
        ax_inputs = np.random.binomial(1, self.p_noise, (n, self.n_bits * self.n_sim))

        # Activate whenever every target are active
        for i in range(n):
            # Build next output
            for k, l_indices in enumerate(self.target_bits):
                if ax_inputs[i, l_indices].all():
                    ax_outputs[i, k] = 1

        return ax_inputs, ax_outputs

    def generate_target_sequence(self, n):
        ax_inputs, ax_outputs = np.zeros((n, self.n_bits * self.n_sim)), np.zeros((n, self.n_sim))
        ax_activations = np.random.binomial(1, self.p_target, (n, self.n_sim))

        for i in range(n):
            for j in range(self.n_sim):
                if ax_activations[i, j] == 1:
                    ax_inputs[i, self.target_bits[j]], ax_outputs[i, j] = 1, 1

        return ax_inputs, ax_outputs


class SparseActivationGrid(GridSim):

    def __init__(self, p_targets, p_bits, n_targets, n_bits, purity_rank=2, delta=0):

        # Size of the simulation
        self.n_bits, self.n_targets = n_bits, n_targets

        # Base param of simulation
        self.p_targets, self.p_bits, self.purity_rank, self.delta = p_targets, p_bits, purity_rank, delta
        self._omega = 0
        self.p, self.q = 1, 1

        # Init mapping targets to bits
        self.map_targets_bits = {'target_{}'.format(j): [] for j in range(self.n_targets)}

        GridSim.__init__(self, 'SparseActivationGrid')

    def phi(self, omega):
        return self.p_targets / (self.p_targets + (1 - self.p_targets) * omega)

    def mean_score(self, t, i, purity=None):
        return self.N(t, i) + \
               int(t * (self.phi(self.omega(i, purity=purity)) * (self.p + self.q) - self.p))

    def omega(self, i, purity=None):

        if purity is None:
            purity = self.purity_rank

        assert purity >= 1

        if i > 0:
            return self._omega - self.delta

        return 1 - pow(1 - self.p_targets, purity - 1)

    def N(self, t, i):
        return - int(t * (self.phi(self.omega(i)) * (self.p + self.q) - self.p))

    def build_map_targets_bits(self,):
        for k in self.map_targets_bits.keys():
            ax_mask = np.random.binomial(1, self.p_bits, self.n_bits).astype(bool)
            self.map_targets_bits[k] = np.arange(self.n_bits)[ax_mask]

        return self

    def set_score_params(self, i):
        self.p = np.ceil(self.phi(self.omega(i)) / (1 - self.phi(self.omega(i))))
        return self

    def get_ranks(self, key):

        d_score = {i: 0 for i in self.map_targets_bits['target_{}'.format(key)]}

        for v in self.map_targets_bits.values():
            for i in v:
                if d_score.get(i, -1) >= 0:
                    d_score[i] += 1

        return d_score

    def estimate_omega(self, l_bits, target, n_sample):

        sax_inputs, sax_outputs = self.generate_io_sequence(n_sample)

        # Get input and output of interest
        ax_output = sax_outputs.toarray()[:, target]
        ax_input = (sax_inputs[:, l_bits].toarray().sum(axis=1) >= len(l_bits)).astype(int)

        # Get precision and omega
        phi = float(ax_input.dot(ax_output)) / ax_input.sum()
        self._omega = self.p_targets * (1 - phi) / ((1 - self.p_targets) * phi)

    def generate_io_sequence(self, n, mask_target=None):

        ax_outputs = np.random.binomial(1, self.p_targets, (n, self.n_targets))
        ax_inputs = self.generate_bit_sequences(ax_outputs)

        if mask_target is not None:
            return csc_matrix(ax_inputs, dtype=bool), csc_matrix(ax_outputs[:, [mask_target]], dtype=bool)

        return csc_matrix(ax_inputs, dtype=bool), csc_matrix(ax_outputs, dtype=bool)

    def generate_bit_sequences(self, ax_activations):

        ax_inputs = np.zeros((ax_activations.shape[0], self.n_bits))
        for i, ax_activation in enumerate(ax_activations):
            for j, activation in enumerate(ax_activation):
                if activation > 0:
                    ax_inputs[i, self.map_targets_bits['target_{}'.format(j)]] = 1

        return ax_inputs