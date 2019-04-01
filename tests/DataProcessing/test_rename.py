# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
test_rename.py
Created by lex at 2019-04-01.
"""

from NetEmbs.DataProcessing.rename_columns import rename_columns
from NetEmbs.GenerateData.complex_df import sales_collections


def test_rename():
    df = sales_collections(titles=["transactionID", "accountID", "Journal", "Date", "debitAmount", "creditAmount"])
    df = rename_columns(df)
    assert list(df) == ["ID", "FA_Name", "Journal", "Date", "Debit", "Credit"]
