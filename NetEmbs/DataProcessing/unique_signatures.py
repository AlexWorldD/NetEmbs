# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
unique_signatures.py
Created by lex at 2019-03-28.
"""
import pandas as pd
from NetEmbs.CONFIG import N_DIGITS


def get_signature(df):
    """
    Aggregation function over GroupBy object: to extract unique signature for the given business process
    :param df: Unique business process as GroupBy DataFrame
    :return: Pandas Series with ID and Signature
    If business process includes only 1-1 flow (e.g. from Cash to Tax), used amount value.
    If business process includes more than 2 transactions, used Credit/Debit values respectfully.
    """
    if df.shape[0] == 2:
        signatureL = list(
            zip(df["FA_Name"][df["Credit"] > 0.0].values, df["amount"][df["Credit"] > 0.0].values.round(N_DIGITS)))
        signatureR = list(
            zip(df["FA_Name"][df["Debit"] > 0.0].values, df["amount"][df["Debit"] > 0.0].values.round(N_DIGITS)))
    elif df.shape[0] > 2:
        # Business process includes more that 2 transactions, hence, can use relative amount for creation signature
        signatureL = sorted(
            list(
                zip(df["FA_Name"][df["Credit"] > 0.0].values, df["Credit"][df["Credit"] > 0.0].values.round(N_DIGITS))),
            key=lambda x: x[0])
        signatureR = sorted(
            list(zip(df["FA_Name"][df["Debit"] > 0.0].values, df["Debit"][df["Debit"] > 0.0].values.round(N_DIGITS))),
            key=lambda x: x[0])
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
