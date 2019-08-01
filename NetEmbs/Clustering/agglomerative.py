# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
agglomerative.py
Created by lex at 2019-05-03.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from NetEmbs.Clustering.find_optimal import find_optimal_nClusters
import logging
from typing import Optional


def cl_Agglomerative(df: pd.DataFrame, n_cl: Optional[int] = 11) -> pd.DataFrame:
    """
    Apply Aglomerative clustering to 'Emb' column
    Parameters
    ----------
    df : DataFrame with column 'Emb'
    n_cl : int, default is 11
        Number of clusters to be found.
        If n_cl=None, then use Silhouette score to find the optimal.

    Returns
    -------
    DataFrame with column 'label'
    """
    if n_cl is None:
        n_cl = find_optimal_nClusters(df, AgglomerativeClustering)
    embdf = pd.DataFrame(list(map(np.ravel, df["Emb"])))
    #     Clustering stuff
    with_cl = df.copy()
    agg = AgglomerativeClustering(n_clusters=n_cl)
    predicted_labels = agg.fit_predict(embdf)
    with_cl["label"] = pd.Series(predicted_labels)
    logging.getLogger(f"{__name__}").info("Agglomerative clustering - DONE")
    return with_cl
