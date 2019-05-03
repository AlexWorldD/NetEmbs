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
from NetEmbs.DataProcessing.stats import getHistCounts
import pandas as pd
from NetEmbs.FSN.graph import FSN


def plotFSN(input_data, colors=("Red", "Blue"), edge_labels=False, node_labels=True, title=None, text_size=16):
    """
    Plot FSN with matplotlib library
    :param fsn: FSN to be visualize
    :param colors: array of colors for FA and BP respectively
    :param edge_labels: True: Show the weights of edges, False: Without the weights of edges, string "NodeName" - only part of edges from that NodeName
    :param title: Title for file to be saved in /img folder. None: no savings
    """
    # Check the input argument type: FSN or DataFrame
    if isinstance(input_data, pd.DataFrame):
        # #     Construct FSN object from the given df
        fsn = FSN()
        fsn.build(input_data, left_title="FA_Name")
        fsn.nodes()
    elif isinstance(input_data, FSN):
        fsn = input_data
    else:
        raise ValueError(
            "Plotting is possible only for DataFrame with journal entries of FSN object! Was given {!s}!".format(
                type(input_data)))
    left = fsn.get_FA()
    pos = nx.bipartite_layout(fsn, left)
    arc_weight = nx.get_edge_attributes(fsn, 'weight')
    node_col = [colors[d['bipartite']] for n, d in fsn.nodes(data=True)]
    BPs = [node for node, d in fsn.nodes(data=True) if d["bipartite"] == 0]
    FAs = [node for node, d in fsn.nodes(data=True) if d["bipartite"] == 1]
    nx.draw_networkx_nodes(fsn, pos, nodelist=BPs, node_shape="D", node_color=colors[1], with_labels=False,
                           node_size=250)
    nx.draw_networkx_nodes(fsn, pos, nodelist=FAs, node_color=colors[0], with_labels=False, node_size=250)
    #     nx.draw_networkx_nodes(fsn, pos, node_color=node_col, with_labels=False, node_size=250)
    debit = {(u, v) for u, v, d in fsn.edges(data=True) if d['type'] == "DEBIT"}
    credit = {(u, v) for u, v, d in fsn.edges(data=True) if d['type'] == "CREDIT"}
    nx.draw_networkx_edges(fsn, pos, edgelist=debit, edge_color="forestgreen", arrowsize=30)
    nx.draw_networkx_edges(fsn, pos, edgelist=credit, edge_color="salmon", arrowsize=30)
    if isinstance(edge_labels, str):
        lbls = {(u, v) for u, v, d in fsn.edges(data=True) if u == edge_labels}
        wei = {item: arc_weight[item] for item in lbls}
        nx.draw_networkx_edge_labels(fsn, pos, node_size=250, edge_labels=wei, font_size=16)
    if edge_labels and isinstance(edge_labels, bool):
        nx.draw_networkx_edge_labels(fsn, pos, node_size=250, edge_labels=arc_weight, font_size=16)
    if node_labels:
        #     TODO add relative align for labels
        label_pos = pos.copy()
        for p in label_pos:  # raise text positions
            label_pos[p][1] += 0.05
        nx.draw_networkx_labels(fsn, label_pos, font_size=text_size)
    ax = plt.gca()
    ax.set_axis_off()
    if title is not None and isinstance(title, str):
        plt.tight_layout()
        plt.savefig("img/" + title, dpi=140, pad_inches=0.01)
    plt.show()


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
    stat_here = getHistCounts(df)
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


def plot_tSNE(fsn_embs, title="tSNE", legend_title="GroundTruth", rand_state=1):
    import os
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    tsne = TSNE(random_state=rand_state)
    embdf = pd.DataFrame(list(map(np.ravel, fsn_embs.iloc[:, 1])))
    embed_tsne = tsne.fit_transform(embdf)
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

    if title is not None and isinstance(title, str):
        plt.tight_layout()
        plt.savefig("img/" + title, dpi=140, pad_inches=0.01)
    plt.show()
