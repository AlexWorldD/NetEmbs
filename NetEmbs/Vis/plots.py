# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
plots.py
Created by lex at 2019-03-15.
"""
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from sklearn import preprocessing
from NetEmbs.DataProcessing.stats.count_financial_accounts import get_hist_counts
from NetEmbs.Vis.helpers import save_to_file
import pandas as pd
from NetEmbs import CONFIG
from NetEmbs.utils.dimensionality_reduction import dim_reduction
from NetEmbs.FSN.graph import FSN
from NetEmbs.Vis.helpers import getColors_Markers
from NetEmbs.utils.evaluation import v_measure
from typing import Tuple, Optional

plt.rcParams["figure.figsize"] = CONFIG.FIG_SIZE

context_settings = {"paper_half": dict(context="paper", font_scale=1.5),
                    "paper_full": dict(context="paper", font_scale=1.8),
                    "talk_half": dict(context="paper", font_scale=3.5),
                    "talk_full": dict(context="talk", font_scale=2)}


def draw_fsn(fsn: FSN, ax=None, colors: Optional[Tuple[str, str]] = ("Red", "Blue"),
             add_edge_labels: Optional[bool] = False,
             add_node_labels: Optional[bool] = True,
             text_size: Optional[int] = 12):
    """
    Draw FSN object as bipartite networks with matplotlib

    Parameters
    ----------
    fsn : FSN
        Input FSn object
    ax : Matplotlib Axes object, optional, default is None
        Draw the graph in the specified Matplotlib axes
    colors : Tuple of colors to fill FA and BP nodes, optional, default is ("Red", "Blue")
    add_edge_labels : bool, optional, default is False
        Add labels for edges to the plot
    add_node_labels : bool, optional, default is True
        Add nodes labels to the plot
    text_size : int, optional, default is 12
        Explicit size for all labels

    Returns
    -------

    """
    if ax is None:
        ax = plt.gca()
    left = fsn.get_FAs()
    pos = nx.bipartite_layout(fsn, left)
    arc_weight = nx.get_edge_attributes(fsn, 'weight')
    nx.draw_networkx_nodes(fsn, pos, ax=ax, nodelist=fsn.get_BPs(), node_shape="D", node_color=colors[1],
                           with_labels=False,
                           node_size=10 * min(5, int(1000 / fsn.number_of_BP())))
    nx.draw_networkx_nodes(fsn, pos, ax=ax, nodelist=fsn.get_FAs(), node_color=colors[0], with_labels=False,
                           node_size=10 * min(5, int(1000 / fsn.number_of_BP())))

    nx.draw_networkx_edges(fsn, pos, edgelist=fsn.get_debit_flows(), edge_color="forestgreen", arrowsize=20)
    nx.draw_networkx_edges(fsn, pos, edgelist=fsn.get_credit_flows(), edge_color="salmon", arrowsize=20)

    if add_edge_labels:
        nx.draw_networkx_edge_labels(fsn, pos, node_size=250, edge_labels=arc_weight, font_size=text_size)
    if add_node_labels:
        label_pos = pos.copy()
        if max(fsn.number_of_FA(), fsn.number_of_BP()) < 8:
            for p in label_pos:  # raise text positions
                label_pos[p][1] += 0.05
        nx.draw_networkx_labels(fsn, label_pos, font_size=text_size)
    ax.set_axis_off()


def plot_financial_accounts_histograms(df: pd.DataFrame, axes: Optional = None, normalized: Optional[bool] = False):
    """
    Plot the distributions of Left/Right-sided Financial accounts in dataset

    Parameters
    ----------
    df : DataFrame
        Input Journal entries
    axes : List of Matplotlib Axes object, optional, default is None
        Draw the graph in the specified Matplotlib axes
    normalized : Bool, optional, default is False
        If True, normalize the histigrams

    Returns
    -------

    """
    from matplotlib.ticker import MaxNLocator
    stat_here = get_hist_counts(df)
    if axes is None:
        fig, axes = plt.subplots(1, 2)
    for i, to_plot in enumerate(stat_here.items()):
        if normalized:
            import numpy as np
            hist = axes[i].bar(to_plot[1].keys(), list(to_plot[1].values()) / np.sum(list(to_plot[1].values())))
        else:
            hist = axes[i].bar(to_plot[1].keys(), to_plot[1].values())
        axes[i].set_xlim((0.5, 10.5))
        axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[i].set_title(to_plot[0] + "-side number of FAs")
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['top'].set_visible(False)
        axes[i].xaxis.set_ticks_position('bottom')
        axes[i].yaxis.set_ticks_position('left')
    return hist


def draw_embeddings(df: pd.DataFrame, ax: Optional = None, legend_title: Optional[str] = "label",
                    context: Optional[str] = "paper_full", save: Optional[bool] = False, **kwargs):
    """
    Draw embeddings as 2D-scatter plot.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with X, Y columns OR Emb column to be used for scatter plot construction
    ax : Matplotlib Axes object, optional, default is None
        Draw the graph in the specified Matplotlib axes
    legend_title : str, optional, default is 'label'
        Column name to be used for colouring
    context : str, optional, default is 'paper_full'
        String name for context leads to minor changes in font-scale
    save : bool, optional, default is False
        Save scatter plot to file.
    kwargs

    Returns
    -------
    Matplotlib Axes object
    """
    if "x" not in df.columns:
        if "Emb" in df.columns:
            df = dim_reduction(df, n_dim=2)
        else:
            raise ValueError(f"Did not find embeddings column (or X, Y) in the given DataFrame!")
    #     Drawing
    cmap, mmap = getColors_Markers(df[legend_title].unique(), n_colors=10, markers=["o", "v", "s"])
    sns.set_context(**context_settings.get(context))

    if ax is None:
        ax = plt.gca()

    emb_vis = sns.scatterplot(
        x="x", y="y",
        hue=legend_title,
        palette=cmap,
        style=legend_title,
        markers=mmap,
        data=df.sort_values(legend_title),
        alpha=0.75, linewidth=0.5,
        s=100, ax=ax
    )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel("Dimension X")
    ax.yaxis.set_ticks_position('left')
    ax.set_ylabel("Dimension Y")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, markerscale=2)
    if legend_title == "GroundTruth":
        ax.set_title("t-SNE visualisation with coloring based on Ground Truth", y=1.08)
    elif legend_title == "label":
        if "GroundTruth" in list(df):
            v_score = ", \n" if context == "talk_half" or context == "paper_half" else ", "
            v_score += "V-Score is " + str(v_measure(df).round(3))
        else:
            v_score = ""
        ax.set_title("t-SNE visualisation with coloring based on predicted labels" + v_score, y=1.08)
    if save:
        save_to_file(ax, **kwargs)
    return emb_vis
