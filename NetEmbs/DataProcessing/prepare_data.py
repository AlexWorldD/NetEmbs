# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
prepare_data.py
Created by lex at 2019-03-28.
"""
from NetEmbs.DataProcessing.normalize import normalize
from NetEmbs.DataProcessing.splitting import split_to_debit_credit, add_from_column
from NetEmbs.DataProcessing.unique_signatures import unique_BPs
from NetEmbs.DataProcessing.merge_FAs import merge_FAs
from NetEmbs.DataProcessing.cleaning import delStrings, CreditDebit
from NetEmbs.CONFIG import PRINT_STATUS, LOG
import logging


def prepare_data(original_df, split=True, merge_fa=True, add_from=True, norm=True, unique=True):
    """
    General function for data preprocessing
    :param original_df:
    :param split: True if Data has to be split into Credit/Debit columns
    :param add_from: True if Data hasn't "from" column (require for FSN construction)
    :param merge_fa: True if need to merge rows with the same affected FAs
    :param norm: True if Data has to be normalized wrt ID
    :param unique: True if Data has to be filtered wrt to Signatures of BPs
    :return: Transformed DF
    """
    if LOG:
        local_logger = logging.getLogger("NetEmbs.DataProcessing.prepare_data")
    if PRINT_STATUS:
        print("Original shape of DataFrame is ", str(original_df.shape))
    if LOG:
        local_logger.info("Original shape of DataFrame is " + str(original_df.shape))
    # Delete all NaNs and Strings values from "Value" column
    original_df = delStrings(original_df)
    if PRINT_STATUS:
        print("Deleted all NaNs and Strings values from 'Value' column: ", str(original_df.shape))
    if LOG:
        local_logger.info("Deleted all NaNs and Strings values from 'Value' column: " + str(original_df.shape))

    if split and "Debit" not in list(original_df):
        original_df = split_to_debit_credit(original_df)
    #     Simplest way to deal with NA values.
    if PRINT_STATUS:
        print("Before merging FAs columns titles are: ", list(original_df))
    if LOG:
        local_logger.info("Before merging FAs columns titles are: " + str(list(original_df)))
    if merge_fa:
        original_df = merge_FAs(original_df)
    if PRINT_STATUS:
        print("After merging FAs columns titles are: ", list(original_df), " shape is ", str(original_df.shape))
    if LOG:
        local_logger.info(
            "After merging FAs columns titles are: " + str(list(original_df)) + " shape is " + str(original_df.shape))
    if add_from:
        original_df = add_from_column(original_df)
    if norm:
        original_df = normalize(original_df)
    #     Remove rows with NaN values after normalization (e.g. when all values were 0.0 -> something/zero leads to NaN)
    original_df.dropna(subset=["Debit", "Credit"], inplace=True)
    if PRINT_STATUS:
        print("After normalization shape of DataFrame is ", str(original_df.shape))
    if LOG:
        local_logger.info("After normalization shape of DataFrame is "+ str(original_df.shape))
    if unique:
        original_df = unique_BPs(original_df)
    if PRINT_STATUS:
        print("Final shape of DataFrame is ", original_df.shape)
    if LOG:
        local_logger.info("Final shape of DataFrame is "+str(original_df.shape))
    return original_df


def prepare_dataMarcel(original_df, merge_fa=True, add_from=True, norm=True, unique=True):
    """
    General function for data preprocessing with given 'type' column
    :param original_df:
    :param add_from: True if Data hasn't "from" column (require for FSN construction)
    :param merge_fa: True if need to merge rows with the same affected FAs
    :param norm: True if Data has to be normalized wrt ID
    :param unique: True if Data has to be filtered wrt to Signatures of BPs
    :return: Transformed DF
    """
    if LOG:
        local_logger = logging.getLogger("NetEmbs.DataProcessing.prepare_data")
    if PRINT_STATUS:
        print("Original shape of DataFrame is ", str(original_df.shape))
    if LOG:
        local_logger.info("Original shape of DataFrame is " + str(original_df.shape))
    # Delete all NaNs and Strings values from "Value" column
    original_df = delStrings(original_df, col_names=["Value"])
    if PRINT_STATUS:
        print("Deleted all NaNs and Strings values from 'Value' column: ", str(original_df.shape))
    if LOG:
        local_logger.info("Deleted all NaNs and Strings values from 'Value' column: " + str(original_df.shape))
    # Splitting into Credit and Debit columns
    original_df = original_df.apply(CreditDebit, axis=1)
    try:
        original_df.drop(columns=["Value", "type"], inplace=True)
    except KeyError:
        raise KeyError("Cannot find 'amount' and 'type' columns in given DataFrame!")
    if PRINT_STATUS:
        print("Splitting into Credit and Debit columns: ", str(original_df.shape))
    if LOG:
        local_logger.info("Splitting into Credit and Debit columns: " + str(original_df.shape))
    if PRINT_STATUS:
        print("Before merging FAs columns titles are: ", list(original_df))
    if LOG:
        local_logger.info("Before merging FAs columns titles are: " + str(list(original_df)))
    #     Simplest way to deal with NA values.
    if merge_fa:
        original_df = merge_FAs(original_df)
    if PRINT_STATUS:
        print("After merging FAs columns titles are: ", list(original_df), " shape is ", str(original_df.shape))
    if LOG:
        local_logger.info(
            "After merging FAs columns titles are: " + str(list(original_df)) + " shape is " + str(original_df.shape))
    if add_from:
        original_df = add_from_column(original_df)
    if norm:
        original_df = normalize(original_df)
    #     Remove rows with NaN values after normalization (e.g. when all values were 0.0 -> something/zero leads to NaN)
    original_df.dropna(subset=["Debit", "Credit"], inplace=True)
    if PRINT_STATUS:
        print("After normalization shape of DataFrame is ", str(original_df.shape))
    if LOG:
        local_logger.info("After normalization shape of DataFrame is "+ str(original_df.shape))
    if unique:
        original_df = unique_BPs(original_df)
    if PRINT_STATUS:
        print("Final shape of DataFrame is ", original_df.shape)
    if LOG:
        local_logger.info("Final shape of DataFrame is "+str(original_df.shape))
    return original_df
