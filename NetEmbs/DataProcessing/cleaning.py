# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
cleaning.py
Created by lex at 2019-05-04.
"""

import pandas as pd
from typing import List


def del_strings(df: pd.DataFrame, cols: List[str] = "Value") -> pd.DataFrame:
    """
    Delete all string values from the given columns
    Parameters
    ----------
    df : Original DataFrame
    cols : str, default is "Value"
            Title/s where to delete all string values

    Returns
    -------
        DataFrame with only numeric values in the given columns
    """
    drop_cl = list()
    for title in cols:
        if title in list(df):
            df[title] = pd.to_numeric(df[title], errors="coerce")
            drop_cl.append(title)
    return df.dropna(subset=drop_cl)
