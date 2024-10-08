import os
import sys

import matplotlib as mpl
import numpy as np
from matplotlib import colors as colors, cm as cmx, pyplot as pl


def set_global_rcParams(rc_config_file):
    """
    Replace the rcParams configuration of matplotlib, with predefined values in the dictionary rc_dict. To restore
    the default values, call mpl.rcdefaults()

    For more details, consult the documentation of rcParams

    :param rc_config_file: path to file containing detailed mpl configuration
    """
    assert os.path.isfile(rc_config_file), 'input must be path to config file'
    mpl.rc_file(rc_config_file)


def get_cmap(N, cmap='hsv'):
    """
    Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.
    """
    color_norm = colors.Normalize(vmin=0, vmax = N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap)

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color


def mark_epochs(ax, epochs, cmap='jet'):
    labels = np.unique(list(epochs.keys()))
    cm = get_cmap(len(labels), cmap)
    for k, v in list(epochs.items()):
        label_index = np.where(k == labels)[0][0]
        ax.fill_betweenx(np.arange(ax.get_ylim()[0], ax.get_ylim()[1], 0.1), v[0], v[1],
                         facecolor=cm(label_index), alpha=0.2)


def fig_output(fig_handle, display=True, save=False):
    """
    Convenience function to display or save a figure
    :param display: bool
    :param save: bool
    :return:
    """
    if display:
        pl.show(block=False)
    if save:
        assert isinstance(save, str), "Please provide filename"
        fig_handle.savefig(save)


def check_axis(ax=None):
    """
    Convenience function to verify if the provided axis is correct and, if no axis is provided, generate one
    :return:
    """
    if (ax is not None) and (not isinstance(ax, mpl.axes.Axes)):
        raise ValueError('ax must be matplotlib.axes.Axes instance.')
    if ax is None:
        fig, ax = pl.subplots()
    else:
        fig = ax.get_figure()
    return fig, ax


def label_bars(rects, ax):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    """
    for rect in rects:
        height = round(rect.get_height(), 2)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def parse_plot_arguments(ax, **kwargs):
    """
    Parse the provided keyword arguments and split them into axis properties and plot properties
    :param ax: matplotlib axis
    :param kwargs:
    :return:
    """
    fig, ax = check_axis(ax)
    if 'suptitle' in kwargs:
        fig.suptitle(kwargs['suptitle'])
        kwargs.pop('suptitle')
    ax_props = {k: v for k, v in kwargs.items() if k in ax.properties()}
    pl_props = {k: v for k, v in kwargs.items() if k not in ax.properties()}
    return ax_props, pl_props


# def progress_bar(progress):
#     """
#     Prints a progress bar to stdout.
#
#     Inputs:
#         progress - a float between 0. and 1.
#
#     Example:
#         >> progress_bar(0.7)
#             |===================================               |
#     """
#     condition_msg = "ERROR: The argument of function visualization.progress_bar(...) must be a float between " \
#                     "0. and 1.!"
#     assert (type(progress) == float) and (progress >= 0.) and (progress <= 1.), condition_msg
#     length = 50
#     filled = int(round(length * progress))
#     sys.stdout.write("\r|" + "=" * filled + " " * (length - filled) + "|")
#     sys.stdout.flush()
