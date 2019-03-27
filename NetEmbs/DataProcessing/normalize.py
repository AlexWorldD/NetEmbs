# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
normalize.py
Last modified by lex at 2019-02-13.
"""
import numpy as np


# TODO try another kind of normalization, unit-normal etc.
def normalize(df, by="ID"):
    dfN = df.copy()
    titles = ["Debit", "Credit"]
    groups = dfN.groupby(by)
    sums = groups[titles].transform(np.sum)
    for column in titles:
        dfN[column] = dfN[column] / sums[column]
    return dfN
