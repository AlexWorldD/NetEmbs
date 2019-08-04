# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
unique_signatures.py
Created by lex at 2019-03-28.
"""
import pandas as pd
from NetEmbs.CONFIG import N_DIGITS


def get_signature(df: pd.DataFrame) -> pd.Series:
    """
    Aggregation function over GroupBy object: to extract unique signature for the given business process.

    If business process includes only 1-1 flow (e.g. from Cash to Tax), used amount value.
    If business process includes more than 2 transactions, used Credit/Debit values respectfully.
    Parameters
    ----------
    df : DataFrame
        Unique business process as GroupBy DataFrame

    Returns
    -------
        Pandas Series with ID and Signature
    """
    signature_l = list()
    signature_r = list()
    if df.shape[0] == 2:
        signature_l = list(
            zip(df["FA_Name"][df["Credit"] > 0.0].values, df["amount"][df["Credit"] > 0.0].values.round(N_DIGITS)))
        signature_r = list(
            zip(df["FA_Name"][df["Debit"] > 0.0].values, df["amount"][df["Debit"] > 0.0].values.round(N_DIGITS)))
    elif df.shape[0] > 2:
        # Business process includes more that 2 transactions, hence, can use relative amount for creation signature
        signature_l = sorted(
            list(
                zip(df["FA_Name"][df["Credit"] > 0.0].values, df["Credit"][df["Credit"] > 0.0].values.round(N_DIGITS))),
            key=lambda x: x[0])
        signature_r = sorted(
            list(zip(df["FA_Name"][df["Debit"] > 0.0].values, df["Debit"][df["Debit"] > 0.0].values.round(N_DIGITS))),
            key=lambda x: x[0])
    return pd.Series({"ID": df["ID"].values[0], "Signature": str((signature_l, signature_r))})


def get_signature_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create DataFrame with ID and Signature
    Parameters
    ----------
    df : DataFrame to be processed

    Returns
    -------
        DataFrame with Signature column
    """
    """
    Helper function for extraction a signature of BP (as a combination of coefficients from left and right part)
    :param original_df:
    :return: DataFrame with BP ID and extracted signature
    """
    res = df.groupby("ID", as_index=False).apply(get_signature)
    return res.drop_duplicates(["Signature"])


def leave_unique_business_processes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtering original DF with respect to unique BP's signatures
    Parameters
    ----------
    df : DataFrame to be processed

    Returns
    -------
        DataFrame with remove duplicated w.r.t. extracted signatures
    """
    signatures = get_signature_df(df)
    return signatures.merge(df, on="ID", how="left")
