# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
rename_columns.py
Created by lex at 2019-04-01.
"""


def rename_columns(df, names={"transactionID": "ID", "accountID": "FA_Name", "debitAmount": "Debit",
                              "creditAmount": "Credit"}):
    """
    Helper function for renaming old columns titles to the default ones suitable for Graph building
    :param df: original DataFrame
    :param names: Dictionary, where key is old name and value is a new name
    {<OldName>: "ID", <OldName>: "FA_Name", <OldName>: "Debit",
                              <OldName>: "Credit"}
    :return: renamed DataFrame
    """
    return df.rename(index=str, columns=names)
