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
import pandas as pd
from NetEmbs import CONFIG
from NetEmbs.utils.dimensionality_reduction import dim_reduction
from NetEmbs.FSN.graph import FSN
from NetEmbs.Vis.helpers import getColors_Markers
from NetEmbs.utils.evaluation import v_measure
from typing import Tuple, Optional

plt.rcParams["figure.figsize"] = CONFIG.FIG_SIZE


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
                           node_size=10*min(5, int(1000/fsn.number_of_BP())))
    nx.draw_networkx_nodes(fsn, pos, ax=ax, nodelist=fsn.get_FAs(), node_color=colors[0], with_labels=False,
                           node_size=10*min(5, int(1000/fsn.number_of_BP())))

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


def plotHeatMap(pairs, title="HeatMap", size=6, norm="col", return_hm=False, absolute_vals=False, debug=False):
    cnt = dict(Counter(pairs))
    heatmap_data = np.zeros((size, size))
    if debug:
        print(cnt)
    for key, item in cnt.items():
        heatmap_data[key] = item
    if norm == "row":
        if not absolute_vals:
            heatmap_data = preprocessing.normalize(heatmap_data, axis=1, norm="l1")
        sns.heatmap(heatmap_data, annot=True, cmap="Blues")
    elif norm == "col":
        if not absolute_vals:
            heatmap_data = preprocessing.normalize(heatmap_data, axis=0, norm="l1")
        mask = np.zeros_like(heatmap_data)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(heatmap_data, mask=mask, annot=True, cmap="Blues")
    if title is not None and isinstance(title, str):
        plt.tight_layout()
        plt.savefig("img/" + title, dpi=140, pad_inches=0.01)
    plt.show()
    if return_hm:
        return heatmap_data


def plotHist(df, title="Histogram", normalized=False):
    """
    Visualize the distribution of the number of FAs involved in BPs
    :param df: Journal entries DataFrame
    :param title: title of img to save
    :param normalized: If True, the histogram height shows a density rather than a count.
    :return:
    """
    stat_here = get_hist_counts(df)
    from matplotlib.ticker import MaxNLocator
    for k, d in stat_here.items():
        ax = plt.figure().gca()
        if normalized:
            import numpy as np
            ax.bar(d.keys(), list(d.values()) / np.sum(list(d.values())))
        else:
            ax.bar(d.keys(), d.values())
        ax.set_xlim((0.5, 10.5))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title(k + "-side number of FAs")
        if title is not None and isinstance(title, str):
            plt.tight_layout()
            plt.savefig("img/" + title + k, dpi=140, pad_inches=0.01)



def plot_tSNE(df, legend_title="label", title="tSNE", folder="", context="paper_full", score=None):
    cmap, mmap = getColors_Markers(df[legend_title].unique(), n_colors=10, markers=["o", "v", "s"])
    dpi = 140
    fig_size = (13, 10)
    #     set_font(14, True)
    if context == "paper_half":
        sns.set_context("paper", font_scale=1.5)
        fig_size = (6.4, 4.8)
    if context == "paper_full":
        sns.set_context("paper", font_scale=1.8)
    if context == "talk_half":
        sns.set_context("paper", font_scale=3.5)
    if context == "talk_full":
        sns.set_context("paper", font_scale=2)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    sns.scatterplot(
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
        v_score = ""
        if context == "talk_half" or context == "paper_half":
            v_score = ", \n"
        else:
            v_score = ", "
        if score is None:
            if "GroundTruth" in list(df):
                v_score += "V-Score is " + str(v_measure(df).round(3))
        else:
            v_score += "V-Score is " + str(np.mean(score).round(3))
        ax.set_title("t-SNE visualisation with coloring based on predicted labels" + v_score, y=1.08)
    if title is not None and isinstance(title, str):
        postfix = "_" + str(CONFIG.STRATEGY) \
                  + "_walks" + str(CONFIG.WALKS_PER_NODE) \
                  + "_pressure" + str(CONFIG.PRESSURE) \
                  + "_EMB" + str(CONFIG.EMBD_SIZE) \
                  + "_TFsteps" + str(CONFIG.STEPS) + "_" + str(CONFIG.EXPERIMENT[0]) + str(CONFIG.EXPERIMENT[1])
        if folder == "":
            fig.savefig(title + "_for_" + context + postfix + ".png", bbox_inches="tight", dpi=dpi,
                        pad_inches=0.05)
        else:
            fig.savefig(folder + "img/" + title + "_for_" + context + postfix + ".png", bbox_inches="tight", dpi=dpi,
                        pad_inches=0.05)
        plt.close(fig)


def plot_PCA(fsn_embs, title="PCA", folder="", legend_title="GroundTruth", rand_state=1):
    import os
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    pca = PCA(n_components=2)
    embdf = pd.DataFrame(list(map(np.ravel, fsn_embs["Emb"])))
    embed_tsne = pca.fit_transform(embdf)
    fsn_embs["x"] = pd.Series(embed_tsne[:, 0])
    fsn_embs["y"] = pd.Series(embed_tsne[:, 1])
    markers = ["o", "v", "s"]
    cur_m = 0
    plt.clf()
    n_gr = 0
    for name, group in fsn_embs.groupby(legend_title):
        n_gr += 1
        if n_gr > 3:
            cur_m = cur_m + 1 if len(markers) - 1 > cur_m else 0
            n_gr = 0
        plt.scatter(group["x"].values, group["y"].values, s=150, marker=markers[cur_m], label=name)
    plt.legend(bbox_to_anchor=(1.3, 1), loc="upper right", frameon=False, markerscale=1)
    if legend_title == "GroundTruth":
        plt.title("Embeddings visualisation with PCA, Ground Truth")
    elif legend_title == "label":
        plt.title("Embeddings visualisation with PCA, predicted labels")

    if title is not None and isinstance(title, str):
        plt.tight_layout()
        postfix = ""
        if folder == "":
            postfix = "_" + "batch" + str(CONFIG.BATCH_SIZE) \
                      + "_emb" + str(CONFIG.EMBD_SIZE) \
                      + "_walks" + str(CONFIG.WALKS_PER_NODE) \
                      + "_TFsteps" + str(CONFIG.STEPS)
        plt.savefig(folder + "img/" + title + postfix, dpi=140, pad_inches=0.01)
