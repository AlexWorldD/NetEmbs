# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
plots.py
Created by lex at 2019-03-15.
"""
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from NetEmbs.DataProcessing.stats.count_financial_accounts import get_hist_counts
from NetEmbs.Vis.helpers import save_to_file
import pandas as pd
from NetEmbs import CONFIG
from NetEmbs.utils.dimensionality_reduction import dim_reduction
from NetEmbs.Vis.plot_vectors import group_embeddings, plot_vectors
from NetEmbs.FSN.graph import FSN
from NetEmbs.Vis.helpers import getColors_Markers, triangle_axis
from NetEmbs.utils.evaluation import v_measure
from typing import Tuple, Optional, List
from wordcloud import WordCloud
import datetime

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
        triangle_axis(axes[i])
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
    triangle_axis(ax)
    ax.set_xlabel("Dimension X")
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


def descriptor_for_cluster(df: pd.DataFrame, grouping_column: Optional[str] = "label",
                           words_column: Optional[str] = "FA_Name", amount_column: Optional[str] = "amount",
                           sort_mode: Optional[str] = "freq",
                           n_top: Optional[int] = 4, save: Optional[bool] = False):
    """
    Show as words cloud the descriptor for the given Journal entries w.r.t. label

    Parameters
    ----------
    df : DataFrame
    grouping_column : str, optional, default is 'label'
    words_column : str, optional, default is 'FA_Name'
        Title of column with information to be used for WordCloud construction
    amount_column : str, optional, default is 'amount'
    sort_mode : str, either 'freq' or 'amount'. Default is 'freq'
    n_top : int, optional, default is 4
        TOP N items to be used for WordCloud construction.
    save : bool, optional, default is False
        Save WordClouds and Histograms to file

    Returns
    -------

    """
    selected_size = df.ID.nunique()

    sns.set_context("paper", font_scale=2.3)
    if grouping_column not in list(df):
        raise KeyError(
            f"Given column name {grouping_column} is not presented in the given DataFrame! Only allows: {list(df)}!")
    if "flow" not in list(df):
        raise KeyError(f"Please ensure that column 'flow' is presented in your DataFrame!")
    if sort_mode not in ["freq", "amount"]:
        raise ValueError(f"Given sort mode is not yet supported. Please use either 'freq' or 'amount' instead!")
    for name, group in df.groupby(grouping_column):
        print(f"Current cluster label is {name}, in selected zone it's "
              f"{round(group.ID.nunique() / selected_size, ) * 100}% of all samples")
        gr = group.groupby([words_column, "flow"])
        counts = gr.size().to_frame(name='counts')
        all_stat = counts.join(gr.agg({amount_column: sum, 'Debit': lambda x: list(x), 'Credit': lambda x: list(x)})
            .rename(
            columns={amount_column: 'amount_sum', 'Debit': 'Debit_list', 'Credit': 'Credit_list'})) \
            .reset_index()
        if sort_mode == "freq":
            all_stat.sort_values(['counts', words_column], ascending=False, inplace=True)
        elif sort_mode == "amount":
            all_stat.sort_values(['amount_sum', words_column], ascending=False, inplace=True)
        #             Store all statistict for N_TOP values as dictionary for further visualization
        text = {"Left": [(x[0], x[2], x[3], x[5]) for x in all_stat[all_stat["flow"] == "outflow"].values[:n_top]],
                "Right": [(x[0], x[2], x[3], x[4]) for x in all_stat[all_stat["flow"] == "inflow"].values[:n_top]]}
        i = 0
        fig, axes = plt.subplots(2, 2)
        for key, data in text.items():
            if sort_mode == "freq":
                # Take the most frequent FA names
                to_vis = [(str(item[0]), item[1]) for item in data]
            elif sort_mode == "amount":
                # Take FA with the highest sum amounts
                to_vis = [(str(item[0]), item[2]) for item in data]
            axes[0, i].set_title(key, size=24)
            wc = WordCloud(background_color="white", width=800, height=400, max_font_size=84, min_font_size=14,
                           repeat=False, relative_scaling=0.8, max_words=100)
            if len(to_vis) > 0:
                wc.generate_from_frequencies(dict(to_vis))
            else:
                continue
            axes[0, i].axis("off")
            axes[0, i].imshow(wc, interpolation="bilinear")
            # Histogram
            [sns.distplot(item[3], label=item[0], kde=False, bins=50, ax=axes[1, i], hist_kws={"range": (0, 1.0)})
             for item in data if len(item[3]) > 10]
            axes[1, i].legend(frameon=False)
            triangle_axis(axes[1, i])
            axes[1, i].set_xlim((0, 1.0))
            i += 1
        if save:
            plt.tight_layout()
            plt.savefig(f"img/WordClouds/Descriptor_for_{grouping_column}={name}_{datetime.datetime.now()}.png",
                        dpi=140, pad_inches=0.01)
