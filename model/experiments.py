#!/usr/bin/env python
# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
experiments.py
Last modified by lex at 2019-02-11.
"""
import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from NetEmbs.DataProcessing import *
from NetEmbs.GenerateData.complex_df import sales_collections
from NetEmbs.FSN.graph import FSN
from NetEmbs.Vis.plots import plotFSN
from NetEmbs.FSN.utils import get_pairs, find_similar

PLOT = False


if __name__ == '__main__':
    d = upload_data(limit=80)
    d = prepare_data(d)
    # fsn = FSN()
    # fsn.build(normalize(d), name_column="FA_Name")
    # fsn.nodes()
    # if PLOT:
    #     plotFSN(fsn, edge_labels=False)
    # t1 = find_similar(d)
    # t2 = find_similar(d, direction="COMBI", column_title="COMBI Similarity")
    t = find_similar(d, direction=["IN", "ALL", "COMBI"])
    print("t")
