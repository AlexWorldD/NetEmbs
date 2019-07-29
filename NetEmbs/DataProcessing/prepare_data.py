# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
prepare_data.py
Created by lex at 2019-03-28.
"""
from NetEmbs.DataProcessing.normalize import normalize
from NetEmbs.DataProcessing.splitting import split_to_debit_credit
from NetEmbs.DataProcessing.unique_signatures import leave_unique_business_processes
from NetEmbs.DataProcessing.merge_FAs import merge_FAs
from NetEmbs.DataProcessing.cleaning import del_strings
import logging

from typing import List
import pandas as pd


def data_preprocessing(df: pd.DataFrame, cols: List[str] = "Value") -> pd.DataFrame:
    """
    Data pre-processing function: clean, normalize and leave unique BPs only
    Parameters
    ----------
    df : DataFrame
        Original DataFrame to be processed
    cols : list of str, default 'Value'
        Titles of columns where delete strings

    Returns
    -------
        DataFrame to be used for FSN construction
    """
    local_logger = logging.getLogger(f"NetEmbs.{__name__}")
    local_logger.info(f"Original shape of DataFrame is {df.shape}")
    # 0. Delete all strings in columns which has to be numeric
    df = del_strings(df, cols)
    local_logger.info(f"Deleted all NaNs and Strings values from 'Value' column: {df.shape}")
    # 1. Add if not exist the Credit/Debit columns
    df = split_to_debit_credit(df)
    # 2. Merge time-split transactions into one
    df = merge_FAs(df)
    local_logger.info(f"After merging FAs the shape is: {df.shape}")
    # 3. Normalize
    df = normalize(df)
    df.dropna(subset=["Debit", "Credit"], inplace=True)
    local_logger.info(f"After normalization the shape is: {df.shape}")
    # 4. Delete duplicated within transactions
    df = leave_unique_business_processes(df)
    # 4. Add 'flow' column if not exist
    df["flow"] = df["Credit"].apply(lambda x: "outflow" if x > 0.0 else "inflow")
    local_logger.info(f"Final shape of DataFrame is: {df.shape}")
    return df
