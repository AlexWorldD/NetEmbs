# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
kmeans.py
Created by lex at 2019-05-03.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from NetEmbs.Clustering.find_optimal import find_optimal_nClusters
from typing import Optional


def cl_KMeans(df: pd.DataFrame, n_cl: Optional[int] = 11) -> pd.DataFrame:
    """
    Apply K-Means clustering to 'Emb' column
    Parameters
    ----------
    df : DataFrame with column 'Emb'
    n_cl : int, default is 11
        Number of clusters to be found.
        If n_cl=None, then use Silhouette score to find the optimal.

    Returns
    -------
    DataFrame
    """
    if n_cl is None:
        n_cl = find_optimal_nClusters(df, KMeans)
    embdf = pd.DataFrame(list(map(np.ravel, df["Emb"])))
    with_cl = df.copy()
    km = KMeans(n_clusters=n_cl)
    predicted_labels = km.fit_predict(embdf)
    with_cl["label"] = pd.Series(predicted_labels)
    return with_cl
