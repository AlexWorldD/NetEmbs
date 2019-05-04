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
        from NetEmbs.DataProcessing.splitting import add_from_column
        self.df = normalize(add_from_column(sales_collections()))
        self.fsn = FSN()
        self.fsn.build(self.df)
        assert len(self.fsn.nodes()) == (self.df['FA_Name'].nunique() + self.df['ID'].nunique())
        assert set(self.fsn.get_FA()) == set(self.df['FA_Name'].unique())
        assert set(self.fsn.get_BP()) == set(self.df['ID'].unique())
        # TODO add test for weights of edges

    def test_projection(self):
        from NetEmbs.FSN.graph import FSN
        from NetEmbs.GenerateData.complex_df import sales_collections
        from NetEmbs.DataProcessing.normalize import normalize
        from NetEmbs.DataProcessing.splitting import add_from_column
        self.df = normalize(add_from_column(sales_collections()))
        self.fsn = FSN()
        self.fsn.build(self.df)
        assert set(self.fsn.projection()) == set(self.df['ID'].unique())
        assert set(self.fsn.projection(on="FA")) == set(self.df['FA_Name'].unique())
