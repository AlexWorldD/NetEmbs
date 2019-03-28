# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
prepare_data.py
Created by lex at 2019-03-28.
"""
from NetEmbs.DataProcessing.normalize import normalize
from NetEmbs.DataProcessing.splitting import split_to_debit_credit
from NetEmbs.DataProcessing.unique_signatures import unique_BPs


def prepare_data(original_df, split=True, norm=True, unique=True):
    """
    General function for data preprocessing
    :param original_df:
    :param split: True if Data has to be split into Credit/Debit columns
    :param norm: True if Data has to be normalized wrt ID
    :param unique: True if Data has to be filtered wrt to Signatures of BPs
    :return: Transformed DF
    """
    if split:
        original_df = split_to_debit_credit(original_df)
    if norm:
        original_df = normalize(original_df)
    if unique:
        original_df = unique_BPs(original_df)
    return original_df
