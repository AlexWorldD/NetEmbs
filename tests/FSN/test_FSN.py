# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
test_FSN.py
Created by lex at 2019-03-14.
"""

import pandas as pd

test_dataset = {'ID': {0: 1, 1: 1, 2: 1, 3: 2, 4: 2},
                'FA_Name': {0: 'Trade Receivables',
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


class TestFSN:
    def test_creation(self):
        from NetEmbs.FSN.graph import FSN
        from NetEmbs.DataProcessing.prepare_data import data_preprocessing
        self.df = data_preprocessing(pd.DataFrame(data=test_dataset))
        self.fsn = FSN()
        self.fsn.build(self.df)
        assert len(self.fsn.nodes()) == (self.df['FA_Name'].nunique() + self.df['ID'].nunique())
        assert set(self.fsn.get_FAs()) == set(self.df['FA_Name'].unique())
        assert set(self.fsn.get_BPs()) == set(self.df['ID'].unique())

    def test_projection(self):
        from NetEmbs.FSN.graph import FSN
        from NetEmbs.DataProcessing.prepare_data import data_preprocessing
        self.df = data_preprocessing(pd.DataFrame(data=test_dataset))
        self.fsn = FSN()
        self.fsn.build(self.df)
        assert set(self.fsn.projection()) == set(self.df['ID'].unique())
        assert set(self.fsn.projection(on="FA")) == set(self.df['FA_Name'].unique())
