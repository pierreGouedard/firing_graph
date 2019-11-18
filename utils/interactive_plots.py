# Global import
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx


def plot_graph(ax_graph, layout=None, title=''):
    """
    plot simple graph
    :param ax_graph: numpy.ndarray
    :param layout:
    :param title:
    :return:
    """
    nx_graph = nx.from_numpy_array(ax_graph)

    if layout is not None:
        d_pos = {}
        for k, v in layout.items():
            nx.draw_networkx_nodes(nx_graph, pos=v['pos'], nodelist=v['pos'].keys(), node_color=v['color'], title=title)
            d_pos.update(v['pos'])

        nx.draw_networkx_edges(nx_graph, d_pos, alpha=0.5, width=2)

    else:
        nx.draw(nx_graph)

    import IPython
    IPython.embed()
    plt.show()


def multi_line_plot_colored(d_series, title='line_plot', ylab='y', xlab='x', cmap=None):
    """
    Plot multiple line whose values are hosted in d_series.

    :param d_series: dictionary containing color key with pandas.Series or numpy arrays
    :param title: title of the figure
    :return: figure
    :rtype: matplotlib.pyplot.figure
    """
    if cmap is None:
        cmap = {}

    # Create figure
    fig = plt.figure(figsize=(20, 15), dpi=72, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
    plt.style.use('bmh')

    for color, l_series in d_series.items():
        for s_series in l_series:
            plt.plot(s_series, '-', linewidth=2, c=cmap.get(color, color))
    import IPython
    IPython.embed()
    plt.ylabel(ylab, fontdict={'size': 45, 'usetex': True}, **{'usetex': True})
    plt.xlabel(xlab,  fontdict={'size': 45, 'usetex': True}, **{'usetex': True})
    plt.title(title, fontdict={'size': 21, 'usetex': True}, **{'usetex': True})
    plt.tick_params(labelsize=30, pad=6)



    plt.show()