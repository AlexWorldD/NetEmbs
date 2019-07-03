# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
helpers.py
Created by lex at 2019-05-02.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def set_font(s=16, reset=False):
    if reset:
        plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["figure.figsize"] = [20, 10]
    #     plt.rcParams['font.family'] = 'serif'
    #     plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rc('font', size=s)  # controls default text sizes
    plt.rc('axes', titlesize=s)  # fontsize of the axes title
    plt.rc('axes', labelsize=s)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=s - 2)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=s - 2)  # fontsize of the tick labels
    plt.rc('legend', fontsize=s)  # legend fontsize
    plt.rc('figure', titlesize=s)  # fontsize of the figure title


# Transform Matplotlib colormap into plotly colorscale:
import itertools
import matplotlib
import numpy as np


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


def getColors_Markers(keys, cm="tab10", n_color=10, markers=["circle", "diamond", "square"]):
    """
    Construct dictionaries with the given keys and requested colors/markers. Used for deterministic behaviour of vis.
    :param keys: The list of the given keys, e.g. unique labels or values of GroundTruth
    :param cm: Name of colormap, Default is "tab10"
    :param n_color: Number of colors, Default is 10
    :param markers: The list of markers. Used if len(keys)>n_color
    :return: Two dictionaries: with key-color and key-marker mappings.
    """
    keys = sorted(keys)
    color_map = dict(zip(keys, matplotlib_to_plotly(cm, n_color) * (len(keys) // n_color + 1)))
    marker_map = dict(
        zip(keys, list(itertools.chain(*[[m] * n_color for m in markers])) * (len(keys) // (3 * n_color) + 1)))
    return color_map, marker_map
