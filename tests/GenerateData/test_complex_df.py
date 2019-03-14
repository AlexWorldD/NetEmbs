# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
test_complex_df.py
Created by lex at 2019-03-14.
"""
from NetEmbs.GenerateData import sales_collections
import pandas as pd


def test_sales_collections():
    df = sales_collections()
    assert isinstance(df, pd.DataFrame)
