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
from NetEmbs.FSN.utils import get_pairs, find_similar, similar, get_JournalEntries, graph_sampling
from NetEmbs.CONFIG import MAIN_LOGGER
from NetEmbs.Logs.custom_logger import log_me

PLOT = False

import time


# def calc_timing(original_function):
#     def new_function(*args, **kwargs):
#         start = time.time()
#         x = original_function(*args, **kwargs)
#         elapsed = time.time()
#         print("Elapsed Time = {0}".format(elapsed - start))
#         return x
#     return new_function()


def time_calc(fsn, jobs=1, runs=10):
    st = time.time()
    for _ in range(runs):
        get_pairs(fsn, n_jobs=jobs)
    end = time.time()
    print("Elapsed time: ", end - st)


if __name__ == '__main__':
    MAIN_LOGGER = log_me()
    MAIN_LOGGER.info("Started..")

    d = upload_data(limit=80)

    MAIN_LOGGER.info("Ended")

    d = prepare_data(d)
    fsn = FSN()
    fsn.build(normalize(d), name_column="FA_Name")
    fsn.nodes()
    time_calc(fsn, runs=10)
    time_calc(fsn, jobs=2, runs=10)
    # t = get_pairs(fsn)
    # if PLOT:
    #     plotFSN(fsn, edge_labels=False)
    # t1 = find_similar(d)
    # t2 = find_similar(d, direction="COMBI", column_title="COMBI Similarity")
    # res = similar(d)
    print("t")
