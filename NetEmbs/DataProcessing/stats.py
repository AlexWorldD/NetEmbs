# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
stats.py
Created by lex at 2019-04-23.
"""

import pandas as pd
from collections import Counter


def get_left_right(df):
    """
    Helper function for counting left-hand and right-hand account for BP
    :param df: grouped object
    :return: Series with number of FA on the left side and on the right side
    """
    return pd.Series({"Left": df[df["from"] == True].count()[0], "Right": df[df["from"] == False].count()[0]})


def getHistCounts(df):
    stat_here = df.groupby("ID", as_index=False).apply(get_left_right)
    res = dict()
    for n in list(stat_here):
        res[n] = Counter(stat_here[n])
    return res
