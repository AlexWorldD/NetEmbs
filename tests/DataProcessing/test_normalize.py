# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
test_normalize.py
Last modified by lex at 2019-02-13.
"""

import pandas as pd
from NetEmbs.DataProcessing.normalize import normalize


def test_normalize():
    raw_data = pd.read_csv("tests/DataProcessing/data/sample.csv", delimiter=";").fillna(0.0)
    data = normalize(raw_data)
    # Credit-debit equation
    assert data["Debit"].sum() - data["Credit"].sum() < 1e-5
    assert data["Debit"][0] == 1.0
    assert data["Credit"][4] == 1.0
