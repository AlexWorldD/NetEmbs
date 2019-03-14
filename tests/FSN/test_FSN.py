# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
test_FSN.py
Created by lex at 2019-03-14.
"""


class TestFSN:
    def test_creation(self):
        from NetEmbs.FSN.graph import FSN
        from NetEmbs.GenerateData.complex_df import sales_collections
        from NetEmbs.DataProcessing.normalize import normalize
        self.df = normalize(sales_collections())
        self.fsn = FSN()
        self.fsn.build(self.df)
        assert len(self.fsn.nodes()) == (self.df['Name'].nunique() + self.df['ID'].nunique())
        assert set(self.fsn.get_FA()) == set(self.df['Name'].unique())
        assert set(self.fsn.get_BP()) == set(self.df['ID'].unique())
        # TODO add test for weights of edges

    def test_projection(self):
        from NetEmbs.FSN.graph import FSN
        from NetEmbs.GenerateData.complex_df import sales_collections
        from NetEmbs.DataProcessing.normalize import normalize
        self.df = normalize(sales_collections())
        self.fsn = FSN()
        self.fsn.build(self.df)
        assert set(self.fsn.projection()) == set(self.df['ID'].unique())
        assert set(self.fsn.projection(on="FA")) == set(self.df['Name'].unique())
