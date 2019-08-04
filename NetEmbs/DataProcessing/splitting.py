# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
data_processing.py
Created by lex at 2019-03-27.
"""

import pandas as pd
from typing import Optional


def split_to_debit_credit(df: pd.DataFrame, value_col: str = "Value", by: Optional[str] = "type") -> pd.DataFrame:
    """
    Explicitly add Credit/Debit columns to DataFrame
    Parameters
    ----------
    df : DataFrame
            Original DataFrame
    value_col : str, default is 'Value'
            Title of the column with raw values in the given DataFrame
    by : str, default is None
            Title for column with 'type' info

    Returns
    -------
        DataFrame with Credit/Debit columns to be used in FSN construction
    """
    if "Credit" in df.columns and "Debit" in df.columns:
        pass
    else:
        try:
            if by is not None and by in df.columns:
                df = df.apply(split_by_type, by=by, axis=1).drop(columns=[value_col, by])
            else:
                df.loc[:, "Debit"] = df.loc[df[value_col] > 0.0, value_col]
                df.loc[:, "Credit"] = -df.loc[df[value_col] < 0.0, value_col]
        except KeyError as k:
            raise KeyError(f"'Value' column is not in the column names... {k}")
    return df


def split_by_type(row: pd.Series, by: str) -> pd.Series:
    """
    Set values to Credit or Debit column considering the other column, e.g. 'type'
    Parameters
    ----------
    row : Series
    by : str
            Column with additional info about the type of transaction

    Returns
    -------
        Row as Series with Credit/Debit columns
    """
    try:
        row["Credit"] = abs(row["Value"]) if row[by] == "credit" else 0.0
        row["Debit"] = abs(row["Value"]) if row[by] == "debit" else 0.0
    except KeyError:
        raise KeyError("Cannot find 'Value' column in given DataFrame!")
    return row
