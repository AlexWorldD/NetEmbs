# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
normalize.py
Last modified by lex at 2019-02-13.
"""
import numpy as np
import pandas as pd
from typing import Optional, List


def normalize(df: pd.DataFrame, cols_to_norm: List[str] = ("Debit", "Credit"), by: Optional[str] = "ID"):
    """
    Normalize values in the given columns
    Parameters
    ----------
    df : DataFrame to be processed
    cols_to_norm : list with str
            Titles for columns to be normalized
    by : str, default is 'ID'
            Title to be used as aggregation key

    Returns
    -------
        DataFrame with normalized values for chosen columns
    """
    df_norm = df.copy()
    df_norm["amount"] = df_norm.apply(lambda row: max(row[t] for t in cols_to_norm), axis=1)
    groups = df_norm.groupby(by)
    sums = groups[cols_to_norm].transform(np.sum)
    for column in cols_to_norm:
        try:
            df_norm[column] = df_norm[column] / sums[column]
        except ZeroDivisionError as e:
            raise ZeroDivisionError(f"Could not normalize the given DataFrame, current row is {sums[column]}")
    return df_norm
