# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
test_normalize.py
Last modified by lex at 2019-02-13.
"""

import pandas as pd
from NetEmbs.DataProcessing.normalize import normalize

test_dataset = {'ID': {0: 1, 1: 1, 2: 1, 3: 2, 4: 2},
                'Name': {0: 'Trade Receivables',
                         1: 'Revenue',
                         2: 'Tax',
                         3: 'Cost of Sales',
                         4: 'Inventories'},
                'Journal': {0: 'Sales ledger',
                            1: 'Sales ledger',
                            2: 'Sales ledger',
                            3: 'Journal ledger',
                            4: 'Journal ledger'},
                'Date': {0: '01/01/2017',
                         1: '01/01/2017',
                         2: '01/01/2017',
                         3: '01/01/2017',
                         4: '01/01/2017'},
                'Debit': {0: 121.0, 1: 0.0, 2: 0.0, 3: 80.0, 4: 0.0},
                'Credit': {0: 0.0, 1: 100.0, 2: 21.0, 3: 0.0, 4: 80.0}}


def test_normalize():
    raw_data = pd.DataFrame(data=test_dataset)
    data = normalize(raw_data)
    # Credit-debit equation
    assert data["Debit"].sum() - data["Credit"].sum() < 1e-5
    assert data["Debit"][0] == 1.0
    assert data["Credit"][4] == 1.0
