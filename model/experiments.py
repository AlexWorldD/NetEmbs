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
from NetEmbs.DataProcessing.normalize import normalize
from NetEmbs.DataProcessing.connect_db import *
from NetEmbs.GenerateData.complex_df import sales_collections
from NetEmbs.FSN.graph import FSN
from NetEmbs.Vis.plots import plotFSN
from NetEmbs.FSN.utils import get_pairs
PLOT = False

if __name__ == '__main__':
    d = upload_data(limit=40)
    fsn = FSN()
    fsn.build(normalize(d), name_column="FA_Name")
    fsn.nodes()
    if PLOT:
        plotFSN(fsn, edge_labels=False)
    get_pairs(fsn)
