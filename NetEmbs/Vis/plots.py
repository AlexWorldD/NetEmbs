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


def plotFSN(fsn, colors=("Red", "Blue"), edge_labels=False, node_labels=True, title=None):
    """
    Plot FSN with matplotlib library
    :param fsn: FSN to be visualize
    :param colors: array of colors for FA and BP respectively
    :param edge_labels: True: Show the weights of edges, False: Without the weights of edges
    :param title: Title for file to be saved in /img folder. None: no savings
    """
    left = fsn.get_FA()
    pos = nx.bipartite_layout(fsn, left)
    arc_weight = nx.get_edge_attributes(fsn, 'weight')
    node_col = [colors[d['bipartite']] for n, d in fsn.nodes(data=True)]
    BPs = [node for node, d in fsn.nodes(data=True) if d["bipartite"] == 0]
    FAs = [node for node, d in fsn.nodes(data=True) if d["bipartite"] == 1]
    nx.draw_networkx_nodes(fsn, pos, nodelist=BPs, node_shape="D", node_color=colors[1], with_labels=False, node_size=250)
    nx.draw_networkx_nodes(fsn, pos, nodelist=FAs, node_color=colors[0], with_labels=False, node_size=250)
#     nx.draw_networkx_nodes(fsn, pos, node_color=node_col, with_labels=False, node_size=250)
    debit = {(u, v) for u, v, d in fsn.edges(data=True) if d['type'] == "DEBIT"}
    credit = {(u, v) for u, v, d in fsn.edges(data=True) if d['type'] == "CREDIT"}
    nx.draw_networkx_edges(fsn, pos, edgelist=debit, edge_color="forestgreen", arrowsize=30)
    nx.draw_networkx_edges(fsn, pos, edgelist=credit, edge_color="salmon", arrowsize=30)
    if edge_labels:
        nx.draw_networkx_edge_labels(fsn, pos, node_size=250, edge_labels=arc_weight, font_size=16)
    if node_labels:
        #     TODO add relative align for labels
        label_pos = pos.copy()
        for p in label_pos:  # raise text positions
            label_pos[p][1] += 0.05
        nx.draw_networkx_labels(fsn, label_pos, font_size=16)
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
