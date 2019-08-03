# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
helpers.py
Created by lex at 2019-05-02.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import matplotlib
import numpy as np
from typing import Optional
from NetEmbs import CONFIG


def save_to_file(ax, title: Optional[str] = None, folder: Optional[str] = None, **kwargs) -> None:
    """
    Wrapper around default savefig to tackle with different config settings
    Parameters
    ----------
    ax : Matplotlib Axes object
        The specified Matplotlib axes to be saved in file
    title : str, optional, default if None
        File name
    folder : str, optional, default is None
        Path to folder to used for file saving
    kwargs

    Returns
    -------
    None
    """
    dpi = kwargs.get("dpi") or 140
    fig_size = kwargs.get("fig_size") or (13, 10)
    if title is not None:
        postfix = "_" + str(CONFIG.STRATEGY) \
                  + "_walks" + str(CONFIG.WALKS_PER_NODE) \
                  + "_pressure" + str(CONFIG.PRESSURE) \
                  + "_EMB" + str(CONFIG.EMBD_SIZE) \
                  + "_TFsteps" + str(CONFIG.STEPS)
        if folder is None:
            plt.savefig(title + "_for_" + postfix + ".png", bbox_inches="tight", dpi=dpi,
                        pad_inches=0.05)
        else:
            plt.savefig(folder + "img/" + title + "_for_" + postfix + ".png", bbox_inches="tight", dpi=dpi,
                        pad_inches=0.05)


def set_font(s=16, reset=False):
    if reset:
        plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["figure.figsize"] = [20, 10]
    #     plt.rcParams['font.family'] = 'serif'
    #     plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rc('font', size=s)  # controls default text sizes
    plt.rc('axes', titlesize=s)  # fontsize of the axes title
    plt.rc('axes', labelsize=s)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=s)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=s)  # fontsize of the tick labels
    plt.rc('legend', fontsize=s)  # legend fontsize
    plt.rc('figure', titlesize=s)  # fontsize of the figure title


# Transform Matplotlib colormap into plotly colorscale:
def matplotlib_to_plotly(color_map="tab10", pl_entries=10):
    """
    Transform Matplotlib colormap into Plotly colormap
    :param color_map: Original Matplotlib's colormap
    :param pl_entries: number of requested colors
    :return: Plotly color-scale
    """
    cmap = matplotlib.cm.get_cmap(color_map)
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

    return pl_colorscale


def getColors_Markers(keys, cm="tab10", n_colors=10, markers=["circle", "diamond", "square"]):
    """
    Construct dictionaries with the given keys and requested colors/markers. Used for deterministic behaviour of vis.
    :param keys: The list of the given keys, e.g. unique labels or values of GroundTruth
    :param cm: Name of colormap, Default is "tab10"
    :param n_colors: Number of colors, Default is 10
    :param markers: The list of markers. Used if len(keys)>n_color
    :return: Two dictionaries: with key-color and key-marker mappings.
    """
    keys = sorted(keys)
    color_map = dict(zip(keys, sns.color_palette(cm, n_colors) * (len(keys) // n_colors + 1)))
    marker_map = dict(
        zip(keys,
            list(itertools.chain(*[[m] * n_colors for m in markers])) * (len(keys) // (len(markers) * n_colors) + 1)))
    return color_map, marker_map
