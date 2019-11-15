# Global imports
from scipy.sparse import csc_matrix
import numpy as np

# Local import
from core.imputer import DoubleArrayImputer
from utils.nmp import NumpyDriver


class TestSignal(object):
    driver = NumpyDriver()

    def __init__(self, name):
        self.name = name
        self.imputer = None

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

    def stream_io_sequence(self, n, mask_target=None, return_dirs=True):
        sax_in, sax_out = self.generate_io_sequence(n, mask_target=mask_target)

        if return_dirs:
            self.imputer, tmpdiri, tmpdiro = self.create_imputer(sax_in, sax_out, return_dirs=return_dirs)
            return self.imputer, tmpdiri, tmpdiro

        else:
            self.imputer = self.create_imputer(sax_in, sax_out)

        return self.imputer

    def create_imputer(self, sax_in, sax_out, return_dirs=False):

        dirname = 'tmp_{}_'.format(self.name)
        tmpdiri = self.driver.TempDir(dirname, suffix='in', create=True)
        tmpdiro = self.driver.TempDir(dirname, suffix='out', create=True)

        # Create I/O and save it into tmpdir files
        self.driver.write_file(sax_in, self.driver.join(tmpdiri.path, 'forward.npz'), is_sparse=True)
        self.driver.write_file(sax_out, self.driver.join(tmpdiri.path, 'backward.npz'), is_sparse=True)

        # Create and init imputer
        imputer = DoubleArrayImputer('test', tmpdiri.path, tmpdiro.path)
        imputer.read_raw_data('forward.npz', 'backward.npz')
        imputer.run_preprocessing()
        imputer.write_features('forward.npz', 'backward.npz')
        imputer.stream_features()

        if return_dirs:
            return imputer, tmpdiri, tmpdiro

        tmpdiri.remove(), tmpdiro.remove()

        return imputer


class SignalPlusNoise(TestSignal):

    def __init__(self, n_sim, n_bits, p_target, n_targets, p_noise):

        # Size of the simulation
        self.n_sim, self.n_bits, self.n_targets = n_sim, n_bits, n_targets

        # Base param of simulation
        self.p_target, self.p_noise = p_target, p_noise
        self.target_bits = [np.random.choice(range(self.n_bits), n_targets, replace=False) for _ in range(self.n_sim)]

        # Set base values of p and q, used in score process
        self.p, self.q = 1, 1

        TestSignal.__init__(self, 'SignalPlusNoise')

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


class SparseActivation(TestSignal):

    def __init__(self, p_targets, p_bits, n_targets, n_bits, purity_rank=2, delta=0):

        # Size of the simulation
        self.n_bits, self.n_targets = n_bits, n_targets

        # Base param of simulation
        self.p_targets, self.p_bits, self.purity_rank, self.delta = p_targets, p_bits, purity_rank, delta
        self._omega = 0
        self.p, self.q = 1, 1

        # Init mapping targets to bits
        self.map_targets_bits = {'target_{}'.format(j): [] for j in range(self.n_targets)}

        TestSignal.__init__(self, 'SparseActivation')

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