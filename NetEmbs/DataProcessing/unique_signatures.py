# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
unique_signatures.py
Created by lex at 2019-03-28.
"""
import pandas as pd


def get_signature(df):
    signatureL = list(zip(df["FA_Name"][df["Value"] < 0.0].values, df["Credit"][df["Value"] < 0.0].values))
    signatureR = list(zip(df["FA_Name"][df["Value"] > 0.0].values, df["Debit"][df["Value"] > 0.0].values))
    return pd.Series({"ID": df["ID"].values[0], "Signature": str((signatureL, signatureR))})


def get_signature_df(original_df):
    """
    Helper function for extraction a signature of BP (as a combination of coeffs from left and right part)
    :param original_df:
    :return: DataFrame with BP ID and extracted signature
    """
    res = original_df.groupby("ID", as_index=False).apply(get_signature)
    return res.drop_duplicates(["Signature"])


def unique_BPs(original_df):
    """
    Filtering original DF with respect to unique BP's signatures
    :param original_df:
    :return:
    """
    signatures = get_signature_df(original_df)
    return signatures.merge(original_df, on="ID", how="left")
