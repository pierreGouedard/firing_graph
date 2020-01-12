# Global import
import numpy as np

# local external import
from ..utils import interactive_plots as ip

# Local import
from .comon import run_signal_plus_noise_simulation

"""
This script is meant to validate that simulated data meets theoretical expectation. More specifically, this script will 
run the simulations of the 'Signal plus Noise' model.
"""

# Global parameter
t, n_bits, p_noise, p_target, n_targets, n_selected = 500, 1000, 0.6, 0.3, 50, 5

ax_noisy_bits, ax_target_bits, simu = run_signal_plus_noise_simulation(
    t, n_bits, p_noise, p_target, n_targets, n_selected, verbose=1
)


# Plot result of optimisation method
ip.multi_line_plot_colored(
    {'b': ax_target_bits[:, 1:], 'r': ax_noisy_bits[:, 1:],
     'k': np.ones((1, ax_target_bits.shape[1])) * simu.mean_score_signal(t, 5)},
    title=r'',
    ylab=r'Value of the score process',
    xlab=r"Number of iteration ($\times 10$)"
)






