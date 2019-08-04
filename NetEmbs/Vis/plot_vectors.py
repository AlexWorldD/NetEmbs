# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
plot_vectors.py
Created by lex at 2019-05-09.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np
from typing import Optional, List
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def group_embeddings(df: pd.DataFrame, how: Optional[str] = "median",
                     by: Optional[str] = "GroundTruth", subset: Optional[List[str]] = None,
                     samples_per_group: Optional[int] = 11) -> pd.DataFrame:
    """
    Helper function to group embeddings w.r.t. the centers of group.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with 'Emb' column to be processed
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
    DataFrame where rows represent the embeddings
    """
    means = dict()
    pretty_vectors = pd.DataFrame(columns=list(df))
    emb_size = df.Emb.head(1).values[0].shape[0]
    for name, group in df.groupby(by):
        if subset is None or (isinstance(subset, list) and name in subset):
            if group.shape[0] > samples_per_group:
                cur_data = group.copy()
                if how == "mean":
                    means[name] = np.mean(group["Emb"].values, axis=0)
                elif how == "median":
                    means[name] = np.median(group["Emb"].values.tolist(), axis=0)
                elif how == "random":
                    means[name] = group["Emb"].values[0]
                cur_data["Similarity"] = cur_data["Emb"].apply(lambda x:
                                                               cosine_similarity(X=means[name].reshape(1, -1),
                                                                                 Y=x.reshape(1, -1)))
                cur_data.sort_values("Similarity", ascending=False, axis=0, inplace=True)
                app = cur_data.head(samples_per_group).copy()
                try:
                    pretty_vectors = pretty_vectors.append(app, sort=True).append(pd.DataFrame(
                        {"ID": 0, "Emb": [[-2] * emb_size], "GroundTruth": None}), sort=False)
                except ValueError:
                    raise ValueError("Could not add empty row to heatmap...")
            else:
                print(f"For group {name} is not enough samples...")
    return pretty_vectors


def plot_vectors(df: pd.DataFrame, ax=None, title: Optional[str] = None, folder: Optional[str] = None):
    """
    Plot embeddings as heat-map.

    That visualisation is common for embeddings, because it allows to see the patterns in the embeddings visually.
    For instance, by GroundTruth grouping, one expects to see the sharp inner patterns.
    In contrast, cross-group patterns should be not too strong.
    Parameters
    ----------
    df : DataFrame
        Input DataFrame with rows as embeddings
    ax : Matplotlib Axes object, optional, default is None
        Draw the graph in the specified Matplotlib axes
    title : str, optional, default if None
        File name
    folder : str, optional, default is None
        Path to folder to used for file saving

    Returns
    -------

    """
    if ax is None:
        ax = plt.gca()
    plt.rc('axes', titlesize=18)  # fontsize of the x and y titles
    plt.rc('axes', labelsize=18)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)  # fontsize of the tick labels
    sns.heatmap(list(df["Emb"].values), ax=ax, vmin=-1.0, vmax=1.0, cmap=sns.color_palette("RdBu_r", 16))
    ax.axes.set_xlabel('')
    ax.axes.set_ylabel('Business processeses')
    ax.axes.set_xlabel('Embedding component')
    ax.axes.xaxis.set_ticklabels(list(range(1, len(df.Emb.values[0]) + 1)))
    ns = np.where(df.GroundTruth.values == None)[0][0]
    ax.axes.yaxis.set_ticklabels(list(df["GroundTruth"].dropna().unique()), rotation='horizontal')
    ax.axes.yaxis.set_major_locator(
        ticker.FixedLocator([ns / 2 + it * (ns + 1) for it in range(df.GroundTruth.dropna().nunique())]))
    plt.tight_layout()
    if title is not None and isinstance(title, str):
        plt.tight_layout()
        postfix = f"_emb_size {len(df['Emb'].values[0])} samples_per_group {ns}"
        if folder is None:
            plt.savefig(title + postfix, dpi=140, pad_inches=0.01)
        else:
            plt.savefig(folder + "img/" + title + postfix, dpi=140, pad_inches=0.01)
