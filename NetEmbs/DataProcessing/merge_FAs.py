# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
merge_FAs.py
Created by lex at 2019-04-12.
"""


def merge_FAs(original_df):
    """
    Merging splitted business processes into one
    :param original_df:
    :return: DataFrame with merged BPs
    """
    titles = list(original_df)
    for t in ["ID", "FA_Name", "Debit", "Credit"]:
        try:
            titles.remove(t)
        except KeyError:
            pass
    agg_dict = {'Credit': 'sum', 'Debit': 'sum'}
    agg_dict.update(dict(zip(titles, ["first"] * len(titles))))
    return original_df.groupby(["ID", "FA_Name"]).agg(agg_dict).reset_index()
