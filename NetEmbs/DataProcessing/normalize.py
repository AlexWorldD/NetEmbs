# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
normalize.py
Last modified by lex at 2019-02-13.
"""
import pandas as pd
import numpy as np


# TODO try another kind of normalization, unit-normal etc.
def normalize(df, by="ID"):
    titles = ["Debit", "Credit"]
    groups = df.groupby(by)
    sums = groups[titles].transform(np.sum)
    for column in titles:
        df[column] = df[column] / sums[column]
    df["from"] = df["Credit"] > 0.0
    return df
