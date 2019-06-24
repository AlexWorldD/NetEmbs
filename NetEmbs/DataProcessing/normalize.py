# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
normalize.py
Last modified by lex at 2019-02-13.
"""
import numpy as np
import pandas as pd
import logging


# TODO try another kind of normalization, unit-normal etc.
def normalize(df, by="ID"):
    dfN = df.copy()
    titles = ["Debit", "Credit"]
    dfN["amount"] = dfN.apply(lambda row: max(row[t] for t in titles), axis=1)
    groups = dfN.groupby(by)
    sums = groups[titles].transform(np.sum)
    for column in titles:
        try:
            dfN[column] = dfN[column] / sums[column]
        except ZeroDivisionError as e:
            logging.exception("Exception occurred", e)
    return dfN
