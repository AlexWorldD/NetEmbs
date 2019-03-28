# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
data_processing.py
Created by lex at 2019-03-27.
"""


def split_to_debit_credit(df):
    """
    Helper function for DataFrame preparation before construction FSN
    :param df: original DataFrame
    :return: FSN-ready DF
    """
    df["Debit"] = df["Value"][df["Value"] > 0.0]
    df["Credit"] = -df["Value"][df["Value"] < 0.0]
    df.fillna(0.0, inplace=True)
    df["from"] = df["Credit"] > 0.0
    return df
