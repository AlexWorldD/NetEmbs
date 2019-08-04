# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
count_financial_accounts.py
Created by lex at 2019-07-29.
"""

import pandas as pd
from collections import Counter
from typing import Dict


def get_left_right(df: pd.DataFrame) -> pd.Series:
    """
        Helper function for counting left-hand and right-hand account for BP
    Parameters
    ----------
    df : DataFrame
        Grouped object as DataFrame

    Returns
    -------
        Series with counters
    """
    return pd.Series({"Left": df[df["flow"] == "inflow"].count()[0], "Right": df[df["flow"] == "outflow"].count()[0]})


def get_hist_counts(df: pd.DataFrame) -> Dict:
    """
    Calculate LH/RH number of FA per each Business process
    Parameters
    ----------
    df : DataFrame
        Original DataFrame to be analysed

    Returns
    -------
        Dictionary with counted values per each Business process ID
    """
    stat_here = df.groupby("ID", as_index=False).apply(get_left_right)
    res = dict()
    for n in list(stat_here):
        res[n] = Counter(stat_here[n])
    return res
