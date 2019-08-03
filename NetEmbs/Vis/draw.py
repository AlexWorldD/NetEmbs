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
from NetEmbs.Vis.plot_vectors import group_embeddings, plot_vectors
from NetEmbs.FSN.graph import FSN
from NetEmbs.Vis.helpers import getColors_Markers
from NetEmbs.utils.evaluation import v_measure
from typing import Tuple, Optional, List

plt.rcParams["figure.figsize"] = CONFIG.FIG_SIZE

context_settings = {"paper_half": dict(context="paper", font_scale=1.5),
                    "paper_full": dict(context="paper", font_scale=1.8),
                    "talk_half": dict(context="paper", font_scale=3.5),
                    "talk_full": dict(context="talk", font_scale=2)}


def fsn(fsn_object: FSN, ax=None, colors: Optional[Tuple[str, str]] = ("Red", "Blue"),
        add_edge_labels: Optional[bool] = False,
        add_node_labels: Optional[bool] = True,
        text_size: Optional[int] = 12):
    """
    Draw FSN object as bipartite networks with matplotlib

    Parameters
    ----------
    fsn_object : FSN
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
    left = fsn_object.get_FAs()
    pos = nx.bipartite_layout(fsn_object, left)
    arc_weight = nx.get_edge_attributes(fsn_object, 'weight')
    nx.draw_networkx_nodes(fsn_object, pos, ax=ax, nodelist=fsn_object.get_BPs(), node_shape="D", node_color=colors[1],
                           with_labels=False,
                           node_size=10 * min(5, int(1000 / fsn_object.number_of_BP())))
    nx.draw_networkx_nodes(fsn_object, pos, ax=ax, nodelist=fsn_object.get_FAs(), node_color=colors[0],
                           with_labels=False,
                           node_size=10 * min(5, int(1000 / fsn_object.number_of_BP())))

    nx.draw_networkx_edges(fsn_object, pos, edgelist=fsn_object.get_debit_flows(), edge_color="forestgreen",
                           arrowsize=20)
    nx.draw_networkx_edges(fsn_object, pos, edgelist=fsn_object.get_credit_flows(), edge_color="salmon", arrowsize=20)

    if add_edge_labels:
        nx.draw_networkx_edge_labels(fsn_object, pos, node_size=250, edge_labels=arc_weight, font_size=text_size)
    if add_node_labels:
        label_pos = pos.copy()
        if max(fsn_object.number_of_FA(), fsn_object.number_of_BP()) < 8:
            for p in label_pos:  # raise text positions
                label_pos[p][1] += 0.05
        nx.draw_networkx_labels(fsn_object, label_pos, font_size=text_size)
    ax.set_axis_off()


def financial_accounts_histograms(df: pd.DataFrame, axes: Optional = None, normalized: Optional[bool] = False):
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


def embeddings_2D(df: pd.DataFrame, ax: Optional = None, legend_title: Optional[str] = "label",
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


def embeddings_as_heatmap(df: pd.DataFrame, title: Optional[str] = None, folder: Optional[str] = None,
                          how: Optional[str] = "median",
                          by: Optional[str] = "GroundTruth", subset: Optional[List[str]] = None,
                          samples_per_group: Optional[int] = 11) -> None:
    """
    Visualise embeddings as heatmap.

    Wrapper around two separate functions: 1) Grouping Embeddings and 2) Vectors visualisation.
    Parameters
    ----------
    df : DataFrame with 'Emb' column
    title : str, optional, default if None
        File name
    folder : str, optional, default is None
        Path to folder to used for file saving
    how : str, optional, default is "median'
        Method to find the center of each group in the embedding space
    by : str, optional, default is GroundTruth
        Column title to be used for grouping
    subset : list of str/int, optional, default is None
        Consider only a subset of values within the chosen column
    samples_per_group : int, optional, default is 11
        Number of samples per group to leave in the output DataFrame

    Returns
    -------
    None
    """
    plot_vectors(group_embeddings(df, how=how, by=by, subset=subset, samples_per_group=samples_per_group),
                 title=title, folder=folder)
