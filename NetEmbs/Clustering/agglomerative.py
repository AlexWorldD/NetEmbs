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


def cl_Agglomerative(df, n_cl=7):
    """
    Clustering in embedding space with KMeans algorithm
    :param df: DataFrame with 'emb' column
    :param n_cl: number of clusters in given data, If None - find the optimal number
    :return:
    """
    if n_cl is None:
        n_cl = find_optimal_nClusters(df, AgglomerativeClustering)
    embdf = pd.DataFrame(list(map(np.ravel, df.iloc[:, 1])))
    #     Clustering stuff
    print("First row of Data: \n", embdf.iloc[0].values)
    agg = AgglomerativeClustering(n_clusters=n_cl)
    predicted_labels = agg.fit_predict(embdf)
    df["label"] = pd.Series(predicted_labels)
    return df
