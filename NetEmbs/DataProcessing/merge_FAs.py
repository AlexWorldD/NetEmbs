# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
merge_FAs.py
Created by lex at 2019-04-12.
"""

import pandas as pd


def merge_FAs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge time-split transactions into one
    Parameters
    ----------
    df : DataFrame
            Original DataFrame to be processed

    Returns
    -------
        DataFrame with merged BPs
    """
    titles = list(df)
    for t in ["ID", "FA_Name", "Debit", "Credit"]:
        try:
            titles.remove(t)
        except KeyError:
            pass
    agg_dict = {'Credit': 'sum', 'Debit': 'sum'}
    agg_dict.update(dict(zip(titles, ["first"] * len(titles))))
    return df.groupby(["ID", "FA_Name"]).agg(agg_dict).reset_index()
