# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
get_centers_per_group.py
Created by lex at 2019-05-10.
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def groupVectors(df, how="median", samples_per_group=11, print_info=False):
    means = dict()
    pretty_vectors = pd.DataFrame(columns=list(df))
    for name, group in df.groupby("GroundTruth"):
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
            if print_info:
                print("-----" + name + "-----")
                print("Highest similarity: \n", cur_data.Similarity.head(2), "\nLowest similarity: \n",
                      cur_data.Similarity.tail(2))
            app = cur_data.head(samples_per_group).copy()
            pretty_vectors = pretty_vectors.append(app, sort=True).append(pd.DataFrame(
                {"ID": 0, "Emb": [[-2] * 32], "GroundTruth": None}), sort=False)
        else:
            print("For group ", name, " is not enough samples...")
    return pretty_vectors
