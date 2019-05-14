# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
data_processing.py
Created by lex at 2019-03-27.
"""


def split_to_debit_credit(df, value_col="Value"):
    """
    Helper function for DataFrame preparation before construction FSN
    :param df: original DataFrame
    :return: FSN-ready DF
    """
    try:
        df.loc[:, "Debit"] = df.loc[df[value_col] > 0.0, value_col]
        df.loc[:, "Credit"] = -df.loc[df[value_col] < 0.0, value_col]
    except KeyError:
        raise KeyError("The Value column is not in the ist... Might be try prepareData(..., split=False, ...) ")
    # df.loc[df[value_col] > 0.0, "Debit"] = df.loc[df[value_col] > 0.0, value_col]
    # df.loc[df[value_col] < 0.0, "Credit"] = -df.loc[df[value_col] < 0.0, value_col]
    return df


def add_from_column(df):
    df["from"] = df["Credit"] > 0.0
    return df
