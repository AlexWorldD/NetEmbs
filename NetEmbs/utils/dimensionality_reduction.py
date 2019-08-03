# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
dimensionality_reduction.py
Created by lex at 2019-07-06.
"""
import pandas as pd
import numpy as np
import os
from sklearn.manifold import TSNE
import logging

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def dim_reduction(df: pd.DataFrame, n_dim: int = 2, rand_state: int = 1):
    """
    Helper function to add X,Y to DataFrame for further visualisation
    Parameters
    ----------
    df : DataFrame
        Input DataFrame with 'Emb' column
    n_dim : int, default is 2
        Number of lower dim space: 2 or 3.
    rand_state : int, default is 1
        Random seed for t-SNE method. To ensure the repeatability.

    Returns
    -------
    Original DataFrame with X,Y (and Z) columns
    """
    if "Emb" in list(df):
        logging.getLogger(f"{__name__}").info("Dimensionality reduction for visualisation purposes")
        if n_dim > 3:
            raise ValueError(
                f"Currently only reduction into 2D/3D space is supported, "
                f"while was given {n_dim} as number of components!")
        tsne = TSNE(n_components=n_dim, random_state=rand_state)
        embdf = pd.DataFrame(list(map(np.ravel, df["Emb"])))
        if embdf.shape[1] > 2:
            embed_tsne = tsne.fit_transform(embdf)
        else:
            embed_tsne = embdf.values
        if n_dim == 2:
            #             2D output space, then set column names as X and Y for visualization purpose
            df["x"] = pd.Series(embed_tsne[:, 0])
            df["y"] = pd.Series(embed_tsne[:, 1])
        elif n_dim == 3:
            #             3D output space, then set column names as X, Y and Z for visualization purpose
            df["x"] = pd.Series(embed_tsne[:, 0])
            df["y"] = pd.Series(embed_tsne[:, 1])
            df["z"] = pd.Series(embed_tsne[:, 2])
        return df
    else:
        raise KeyError("No Embs column in the given DataFrame!")
