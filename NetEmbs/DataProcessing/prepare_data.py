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
    if split and "Debit" not in list(original_df):
        original_df = split_to_debit_credit(original_df)
    #     Simplest way to deal with NA values.
    original_df.fillna(0.0, inplace=True)
    if merge_fa:
        original_df = merge_FAs(original_df)
    if add_from:
        original_df = add_from_column(original_df)
    if norm:
        original_df = normalize(original_df)
    if unique:
        original_df = unique_BPs(original_df)
    return original_df
