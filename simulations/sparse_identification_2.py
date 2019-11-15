# Global import
import numpy as np

# Local import
from tests.utils import interactive_plots as ip
from simulations import utils as u

"""
This script is meant to validate that simulated data meets theoretical expectation. More specifically, this script will 
run the simulation of the 'Signal plus Noise' model.
"""

# Pseudo random
np.random.seed(1234)

# Core params of the simulation
p_targets, p_bits, n_targets, n_bits, purity_rank = 0.3, 0.3, 10, 1000, 4

# drainer params range of delta [0.0, 1e-2, 5e-2, 1.5e-1]
t, i, resolution, delta = 500, 5, 10, 0

# Color map rank
d_cmap = {1: "#006600", 2: "#006644", 3: "#004466", 4: "#000066", 5: "#CC6600", 6: "#994C00", 7: "#CC0000", 8: "#990000"}


d_colored_series, d_rank_color = u.run_sparse_simulation(
    t, i, resolution, p_targets, p_bits, n_targets, n_bits, purity_rank, d_cmap, delta=delta
)

ip.multi_line_plot_colored(
    d_colored_series,
    title=r'',
    ylab=r'Value of the score process',
    xlab=r"Number of iteration ($\times 10$)",
    cmap=d_rank_color
)







