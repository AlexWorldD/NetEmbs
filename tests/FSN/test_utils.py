# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
test_utils.py
Created by lex at 2019-04-12.
"""


class TestUtils:
    def test_make_pairs(self):
        from NetEmbs.FSN.utils import make_pairs
        t = [1, 2, 3, 1, 2]
        out = make_pairs(t, window=0)
        assert out == []
        out = make_pairs(t, window=1)
        assert out == [(1, 2), (2, 1), (2, 3), (3, 2), (3, 1), (1, 3), (1, 2), (2, 1)]
        out = make_pairs(t, window=2)
        assert out[:2] == [(1, 2), (1, 3)]

    def test_get_top_similar(self):
        from NetEmbs.FSN.utils import get_top_similar
        t = [[2, 3], [2, 3], [1, 4], [2, 1], [1, 2]]
        out = get_top_similar(t, as_DataFrame=False)
        assert list(out.keys()) == [2, 1]
        assert out[2] == [(3, 2), (1, 1)]
        out = get_top_similar(t)
        import pandas as pd
        assert isinstance(out, pd.DataFrame)
        assert list(out["ID"].values) == [1, 2]
