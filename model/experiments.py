#!/usr/bin/env python
# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
experiments.py
Last modified by lex at 2019-02-11.
"""
import pandas as pd
from NetEmbs.FSN.utils import get_pairs
from utils.Logs import log_me
from NetEmbs.Clustering import *
from sklearn.cluster import KMeans

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

    d = pd.read_pickle("../tmp_embs.pkl")
    find_optimal_nClusters(d, KMeans)
    # fsn = FSN()
    # fsn.build(normalize(d), left_title="FA_Name")
    # fsn.nodes()
    # plotFSN(fsn)
    # plotFSN(1)
    # time_calc(fsn, runs=10)
    # time_calc(fsn, jobs=4, runs=10)
    # t = get_pairs(fsn)
    # if PLOT:
    #     plotFSN(fsn, edge_labels=False)
    # t1 = find_similar(d)
    # t2 = find_similar(d, direction="COMBI", column_title="COMBI Similarity")
    # res = similar(d)
    print("t")
