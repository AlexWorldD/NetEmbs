# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
find_optimal.py
Created by lex at 2019-05-03.
"""

from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from NetEmbs import CONFIG


def find_optimal_nClusters(df: pd.DataFrame, cl_method) -> int:
    """
    Find optimal number of clusters for unsupervised case.
    Parameters
    ----------
    df : DataFrame
        Input DataFrame with obtained embeddings
    cl_method : Clustering method from sklearn.cluster

    Returns
    -------
    Optimal number of clusters w.r.t. the method and given dataset
    """
    cl = pd.DataFrame(list(map(np.ravel, df.iloc[:, 1])))
    print("First row of Data: \n", cl.iloc[0].values)
    cur_score = 0.0
    cur_num_cl = 2
    for cur_cl in range(2, CONFIG.NUM_CL_MAX):
        km = cl_method(n_clusters=cur_cl)
        predicted_labels = km.fit_predict(cl)
        silhouette_avg = silhouette_score(cl, predicted_labels)
        if silhouette_avg >= cur_score:
            cur_score = silhouette_avg
            cur_num_cl = cur_cl
    print(f"Current clustering method is {cl_method}")
    print(f"Optimal number of clusters is {cur_num_cl}. The average silhouette_score is {cur_score}")
    return cur_num_cl
