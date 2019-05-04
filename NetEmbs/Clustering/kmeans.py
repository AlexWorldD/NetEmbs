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


def cl_KMeans(df, n_cl=7):
    """
    Clustering in embedding space with KMeans algorithm
    :param df: DataFrame with 'emb' column
    :param n_cl: number of clusters in given data, If None - find the optimal number
    :return:
    """
    if n_cl is None:
        n_cl = find_optimal_nClusters(df, KMeans)
    embdf = pd.DataFrame(list(map(np.ravel, df["Emb"])))
    #     Clustering stuff
    print("First row of Data: \n", embdf.iloc[0].values)
    km = KMeans(n_clusters=n_cl)
    predicted_labels = km.fit_predict(embdf)
    df["label"] = pd.Series(predicted_labels)
    return df
