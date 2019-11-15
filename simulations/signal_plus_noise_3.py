# Global import
import pickle
import numpy as np

# Local import
from simulations import utils as u

"""
This script is meant to validate that simulated data meets theoretical expectation. More specifically, this script will 
run the simulation of the 'Signal plus Noise' model.
"""

# Global parameter
t, n_bits, n_exp, p_target, n_targets = 200, 1000, 100, 0.3, 50

l_p_noise = [0.3, 0.5, 0.7, 0.9]

# Run simuation
d_results = {}
for p_noise in l_p_noise:
    l_phi, l_psi, n_empty = [], [], 0

    for i in range(n_exp):
        # Run simulation
        ax_noisy_bits, ax_target_bits, simu = u.run_signal_plus_noise_simulation(
            t, n_bits, p_noise, p_target, n_targets, 0, p_sample=0.5
        )

        n_target = len(ax_target_bits[:, -1][ax_target_bits[:, -1] > 0])
        n_noisy = len(ax_noisy_bits[:, -1][ax_noisy_bits[:, -1] > 0])

        if n_noisy + n_target > 0:
            l_phi += [simu.phi((pow(p_noise, n_target)))]
            l_psi += [simu.mu(n_noisy)]
        else:
            n_empty += 1

    if len(l_phi) == 0:
        l_phi, l_psi = [0], [0]

    d_results['noise={}'.format(p_noise)] = {
        'precision_mean': np.mean(l_phi), 'precision_std': np.std(l_phi), 'recall_mean': np.mean(l_psi),
        'recall_std': np.std(l_psi), 'n_empty': n_empty
    }
    with open('res_sim_3.pickle', 'wb') as handle:
        pickle.dump(d_results, handle)

    print('{} experiments finished for p_noise= {}'.format(n_exp, p_noise))

