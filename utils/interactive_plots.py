# Global import
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.pyplot import cm

l_colors = cm.rainbow(np.linspace(0, 1, 5))


def plot_spectogram(Zxx, f, t):
    """
    Interactive plot of spectogram
    :param Zxx: array-like result of the stft
    :param f: array-like frequency range
    :param t: array-like time range
    :return:
    """

    # run code below
    Sxx = np.abs(Zxx)

    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    import IPython
    IPython.embed()
    plt.show()


def plot_signal(s, t):

    plt.plot(t, s)
    plt.ylabel('value')
    plt.xlabel('Time [sec]')
    import IPython
    IPython.embed()
    plt.show()


def plot_signals(l_s, t):

    fig = plt.figure()
    for s in l_s:
        plt.plot(t, s)

    plt.ylabel('value')
    plt.xlabel('Time [sec]')
    import IPython
    IPython.embed()
    plt.show()


def plot_fft(N, xf, yf):
    """
    plot frequencies
    :param N: int len of signal / fft
    :param xf: array-like frequencies (label of x axis)
    :param yf: frequency power
    :return:
    """

    plt.plot(xf, 1.0 / N * np.abs(yf))
    plt.grid()
    import IPython
    IPython.embed()

    plt.show()


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


def plot_hist(ax_values):

    plt.hist(x=ax_values, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid()
    import IPython
    IPython.embed()
    plt.show()


def multi_line_plot_legend(d_series, title='line_plot', lloc='upper left'):
    """
    Plot multiple line whose values are hosted in d_series.

    :param d_series: dictionary containing pandas.Series
    :param title: title of the figure
    :return: figure
    :rtype: matplotlib.pyplot.figure
    """
    # Create figure
    fig = plt.figure(figsize=(20, 15), dpi=72, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))

    for i, (name, s_serie) in enumerate(d_series.items()):
        plt.plot(s_serie, '-o', linewidth=2, label=name, c=l_colors[i % len(l_colors)])

    plt.legend(loc=lloc, prop={'size': 12})
    plt.title(title)

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

    plt.ylabel(ylab, fontdict={'size': 45, 'usetex': True}, **{'usetex': True})
    plt.xlabel(xlab,  fontdict={'size': 45, 'usetex': True}, **{'usetex': True})
    plt.title(title, fontdict={'size': 21, 'usetex': True}, **{'usetex': True})
    plt.tick_params(labelsize=30, pad=6)
    import IPython
    IPython.embed()
    plt.show()


def multi_box_plot(d_series, title='multiple boxplot', ylab='y', xlab='x', name_data='data'):
    """
    Plot multiple line whose values are hosted in d_series.

    :param d_series: dictionary containing color key with pandas.Series or numpy arrays
    :param title: title of the figure
    :return: figure
    :rtype: matplotlib.pyplot.figure
    """
    # Create figure
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 15), dpi=72, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))

    for name, d_box in d_series.items():
        bplot = axes.boxplot(labels=d_box['x'], x=d_box[name_data], patch_artist=True)

        for patch in bplot['boxes']:
            patch.set_facecolor(d_box['color'])

    plt.ylabel(ylab, fontdict={'size': 18, 'usetex': True}, **{'usetex': True})
    plt.xlabel(xlab, fontdict={'size': 18, 'usetex': True}, **{'usetex': True})
    plt.title(title, fontdict={'size': 21, 'usetex': True}, **{'usetex': True})

    import IPython
    IPython.embed()
    plt.show()


def bar_plot(df_hist, l_col_val, title="Histogram", ylab='val', width=0.3, lloc='upper left'):
    """
    Plot (multiple) histogram whose values are hosted in df

    :param df_hist: DataFrame hosting heights of bars
    :type df_hist: DataFrame
    :param l_col_val: list of name of columns of df_hist that host height of bar of the plot
    :type l_col_val: list of str
    :param title: str title of the plot
    :param ylab: str y label
    :param width: float width of the bar < 1.
    :param lloc: str position of the legend

    :return: pyplot.figure
    """
    # Create figure
    fi = plt.figure(figsize=(20, 15), dpi=72, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
    plt.style.use('bmh')

    for i, name in enumerate(l_col_val):
        # Make bar plot with correct ticks
        plt.bar(np.arange(len(df_hist)) + ((i - 1) * width), df_hist[name], width, align='center', alpha=0.5, label=name)
        plt.xticks(np.arange(len(df_hist)), df_hist.index)

    # Add title and ordinate label
    plt.ylabel(ylab)
    plt.legend(loc=lloc, prop={'size': 12})
    plt.title(title)

    return fi


def scatter_plot(df_scat, col_x, col_y, col_color=None, title='Scatter Plot', s=None):
    """
    Ploting 2D scatter plot whose point coordinate are hosted in df.

    :param df_scat: Dataframe hosting coordinates of bars
    :type df_scat: DataFrame
    :param col_x: name of columns that host x coordinate
    :param col_y: name of columns that host y coordinate
    :param col_color: str name of columns that host color of scatter point
    :param title: str title of the figure
    :param s: float size of point
    :return: pyplot.figure
    """
    if s is None:
        s = 20 * 3 ** 4

    fi = plt.figure(figsize=(20, 15), dpi=72, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))

    with plt.style.context('fivethirtyeight'):

        if col_color is not None:
            for i, (color, df_sub) in enumerate(df_scat.groupby([col_color])):
                plt.scatter(df_sub[col_x], df_sub[col_y], c=l_colors[i % len(l_colors)], s=s)

        else:
            plt.scatter(df_scat[col_x], df_scat[col_y], c='b', s=s)

    plt.title(title)

    return fi



