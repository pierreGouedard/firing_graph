# Global import
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
