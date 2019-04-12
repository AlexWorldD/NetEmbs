# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
merge_FAs.py
Created by lex at 2019-04-12.
"""


def merge_FAs(original_df):
    return original_df.groupby(["ID", "FA_Name"]).agg({'Debit': 'sum', 'Credit': 'sum'}).reset_index()
