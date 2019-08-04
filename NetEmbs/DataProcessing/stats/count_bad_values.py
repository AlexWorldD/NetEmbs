# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
count_bad_values.py
Created by lex at 2019-07-29.
"""
import pandas as pd
from typing import List, Dict


def count_strings(df: pd.DataFrame, cols: List[str] = ("Credit", "Debit")) -> Dict[str, int]:
    """
    Count number of string values within given columns
    Parameters
    ----------
    df : DataFrame to be analysed
    cols : list
            Columns' titles to be analyzed
    Returns
    -------
        Dictionary <column title, count values>
    """
    output = dict()
    for title in cols:
        if title in df.columns:
            output[title] = df[title].map(lambda x: 1 if type(x) == str else 0).sum()
    return output


def count_nan(df: pd.DataFrame, cols: List[str] = ("Credit", "Debit")) -> Dict[str, int]:
    """
    Count number of NaN values within given columns
    Parameters
    ----------
    df : DataFrame to be analysed
    cols : list
            Columns' titles to be analyzed
    Returns
    -------
        Dictionary <column title, count values>
    """
    output = dict()
    for title in cols:
        if title in df.columns:
            output[title] = df[title].isnull().sum()
    return output


def count_zero(df: pd.DataFrame, cols: List[str] = ("Credit", "Debit")) -> Dict[str, int]:
    """
    Count number of transaction where 0.0 either for Credit or Debit columns values within given columns
    Parameters
    ----------
    df : DataFrame to be analysed
    cols : list

    Returns
    -------
        Dictionary <column title, count values>
    """
    output = dict()
    try:
        aggregation = df.groupby("ID")[cols].sum()
        for title in cols:
            output[title] = aggregation.loc[aggregation[title] == 0.0, title].count()
    except KeyError as error:
        raise KeyError(f"Given columns titles are not in the DataFrame, {list(df)} are only accepted! \n"
                       f" Error info: {error}")
    return output


def count_bad_values(df: pd.DataFrame, cols: List[str] = ("Credit", "Debit")) -> None:
    """
    Print the number of found errors in the given DataFrame
    Parameters
    ----------
    df : DataFrame to be analysed
    cols : list
            Columns' titles to be analyzed

    Returns
    -------
        None
    """
    print("Strings in numeric columns: ", count_strings(df, cols))
    print("NaN in numeric columns: ", count_nan(df, cols))
    print("Zeros BPs: ", count_zero(df, cols))
