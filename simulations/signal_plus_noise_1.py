# Global import
import numpy as np

# Local import
from tests.utils import interactive_plots as ip
from simulations import utils as u

"""
This script is meant to validate that simulated data meets theoretical expectation. More specifically, this script will 
run the simulation of the 'Signal plus Noise' model.
"""

# Global parameter
t, n_bits, p_noise, p_target, n_targets = 500, 1000, 0.3, 0.3, 50

# Run simulation
ax_noisy_bits, ax_target_bits, simu = u.run_signal_plus_noise_simulation(t, n_bits, p_noise, p_target, n_targets, 0)

# Plot the result
ip.multi_line_plot_colored(
    {'b': ax_target_bits[:, 1:], 'r': ax_noisy_bits[:, 1:],
     'k': np.ones((1, ax_target_bits.shape[1])) * simu.mean_score_signal(t, 0)},
    title=r'',
    ylab=r'Value of the score process',
    xlab=r"Number of iteration ($\times 10$)"
)
