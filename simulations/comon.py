# Global import
import numpy as np

# local external import
from utils import grid_sim as gs
from core.solver import sampler, drainer

# Local import


def run_signal_plus_noise_simulation(t, n_bits, p_noise, p_target, n_targets, i, p_sample=1., resolution=10, verbose=0):
    """

    :param t:
    :param n_bits:
    :param p_noise:
    :param p_target:
    :param n_targets:
    :param i:
    :return:
    """

    # Create simulations and get imputer
    simu = gs.SignalPlusNoiseGrid(1, n_bits, p_target, n_targets, p_noise)
    imputer = simu.stream_io_sequence(10000)
    simu.set_score_params(i)

    if i > 0:
        target_bits = {0: np.random.choice(simu.target_bits[0], i, replace=False)}
    else:
        target_bits = None

    # Sample and drain
    smp = sampler.Sampler(
        [simu.n_sim * simu.n_bits, simu.n_sim], simu.N(t, i), imputer, p_sample=p_sample, selected_bits=target_bits,
    ).sample().build_graph_multiple_output()

    # Drain
    drn = drainer.FiringGraphDrainer(t, simu.p, simu.q, resolution, smp.firing_graph.copy(), imputer, verbose=1)

    # init tracking of simulations
    l_ps_bits, l_target_bits = list(smp.preselect_bits[0]), list(simu.target_bits[0])
    l_noisy_bits = list(set(l_ps_bits).difference(l_target_bits))
    ax_noisy_bits, ax_target_bits = np.zeros((len(l_noisy_bits), 1)), np.zeros((len(l_target_bits), 1))

    # Print some general information on simulations
    if verbose > 0:
        print('simulations parameter: i={}, t={}, N={}, mean_score={}'.format(
            i, t, simu.N(t, i), simu.mean_score_signal(t, i)
        ))

    stop, j = False, 0
    while not stop:
        for _ in range(int(resolution / drn.bs)):
            fg = drn.drain().firing_graph
            drn.reset_all()

        ax_noisy_bits = np.hstack((ax_noisy_bits, fg.Iw.toarray()[l_noisy_bits, 0][:, np.newaxis]))
        ax_target_bits = np.hstack((ax_target_bits, fg.Iw.toarray()[l_target_bits, 0][:, np.newaxis]))

        if not fg.Im.toarray().any():
            stop = True
            continue

        # Adapt batch size according to distance with t
        batch_size = max(min(t - fg.backward_firing['i'].toarray().max(), resolution), 1)
        drn.bs = int(resolution / np.ceil(resolution / batch_size))
        drn.reset_all()
        j += 1

        print('Iteration {}'.format(j * resolution))

    return ax_noisy_bits, ax_target_bits, simu


def run_sparse_simulation(t, i, resolution, p_targets, p_bits, n_targets, n_bits, purity_rank, d_cmap, target=0,
                          delta=0):
    """

    :param t:
    :param n_bits:
    :param p_noise:
    :param p_target:
    :param n_targets:
    :param i:
    :return:
    """
    # Create simulations abd generate I/O
    simu = gs.SparseActivationGrid(p_targets, p_bits, n_targets, n_bits, purity_rank=purity_rank, delta=delta) \
        .build_map_targets_bits()
    imputer = simu.stream_io_sequence(10000, mask_target=target)
    simu.set_score_params(i)

    if i > 0:
        d_rank = simu.get_ranks(target)
        target_bits = {0: np.random.choice([k for k, v in d_rank.items() if v == purity_rank], i, replace=False)}
        simu.estimate_omega(target_bits[target], 0, 5000)
    else:
        target_bits = {}

    # Sample and drain
    smp = sampler.Sampler([simu.n_bits, 1], simu.N(t, i), imputer, selected_bits=target_bits) \
        .sample() \
        .build_graph_multiple_output()

    # Get targets bits and compute their rank
    l_bits = list(smp.preselect_bits[target])
    ax_bits = np.zeros((len(l_bits), 1))
    d_rank = {k: v for k, v in simu.get_ranks(target).items() if k not in target_bits.get(target, {})}

    # Drain
    drn = drainer.FiringGraphDrainer(t, simu.p, simu.q, resolution, smp.firing_graph.copy(), imputer)
    stop, j = False, 0
    while not stop:
        for _ in range(int(resolution / drn.bs)):
            fg = drn.drain().firing_graph
            drn.reset_all()

        ax_bits = np.hstack((ax_bits, fg.Iw.toarray()[l_bits, target][:, np.newaxis]))

        if not fg.Im.toarray().any():
            stop = True
            continue

        # Adapt batch size according to distance with t
        sax_bfi = fg.backward_firing['i'].multiply(fg.Im)
        batch_size = max(min(t - sax_bfi.tocsc().max(), resolution), 1)
        drn.bs = int(resolution / np.ceil(resolution / batch_size))
        drn.reset_all()
        j += 1

        print('Iteration {}'.format(j * resolution))

    # Map score stochastic process by colour (= purity rank)
    d_colored_series, d_rank_color = {}, {}
    for k, v in d_rank.items():
        if v < 1:
            continue

        if v not in d_rank_color.keys():
            c = d_cmap[v]
            d_rank_color[v] = c

        d_colored_series[v] = d_colored_series.get(v, []) + [ax_bits[l_bits.index(k), 1:]]

    return d_colored_series, d_rank_color
