# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
test_pairs_construction.py
Created by lex at 2019-08-03.
"""


class TestSampling:
    def test_pairs_construction(self):
        from NetEmbs.GraphSampling.sampling import pairs_construction
        t = [[1, 2, 3, 1, 2]]
        out = pairs_construction(t, window_size=0, create_folder=False)
        assert out == []
        out = pairs_construction(t, window_size=1, create_folder=False)
        assert out == [(1, 2), (2, 1), (2, 3), (3, 2), (3, 1), (1, 3), (1, 2), (2, 1)]
        out = pairs_construction(t, window_size=2, create_folder=False)
        assert out[:2] == [(1, 2), (1, 3)]
