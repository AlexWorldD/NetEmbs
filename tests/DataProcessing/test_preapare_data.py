# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
test_preapare_data.py
Created by lex at 2019-04-01.
"""

from NetEmbs.DataProcessing.rename_columns import rename_columns
from NetEmbs.DataProcessing.unique_signatures import unique_BPs
from NetEmbs.DataProcessing.splitting import add_from_column
import pandas as pd

test_data = {'transactionID': {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2},
             'accountID': {0: 'Revenue',
                           1: 'Tax',
                           2: 'Trade Receivables',
                           3: 'Trade Receivables',
                           4: 'Cash',
                           5: 'Revenue',
                           6: 'Tax',
                           7: 'Trade Receivables'},
             'Journal': {0: 'Sales ledger',
                         1: 'Sales ledger',
                         2: 'Sales ledger',
                         3: 'Sales ledger',
                         4: 'Sales ledger',
                         5: 'Sales ledger',
                         6: 'Sales ledger',
                         7: 'Sales ledger'},
             'Date': {0: '01/01/2017',
                      1: '01/01/2017',
                      2: '01/01/2017',
                      3: '01/01/2017',
                      4: '01/01/2017',
                      5: '01/01/2017',
                      6: '01/01/2017',
                      7: '01/01/2017'},
             'debitAmount': {0: 0.0,
                             1: 0.0,
                             2: 429.18,
                             3: 0.0,
                             4: 429.18,
                             5: 0.0,
                             6: 0.0,
                             7: 1059.82},
             'creditAmount': {0: 403.0,
                              1: 26.18,
                              2: 0.0,
                              3: 429.18,
                              4: 0.0,
                              5: 997.0,
                              6: 62.82,
                              7: 0.0}}


def test_add_from():
    df = rename_columns(pd.DataFrame(test_data))
    df = add_from_column(df)
    assert max(df['from'].values == (df['Credit'] > 0).values) == True



