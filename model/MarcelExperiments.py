# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
MarcelExperiments.py
Created by lex at 2019-05-05.
"""
from NetEmbs import *
from NetEmbs.Logs.custom_logger import log_me
import pandas as pd
import numpy as np

# TODO here you can select with what kind of data you are going to work: Simulated or Real one
MODE = "SimulatedData"
# MODE = "RealData"

if __name__ == '__main__':
    print("Welcome to NetEmbs application!")
    MAIN_LOGGER = log_me()
    MAIN_LOGGER.info("Started..")
    if MODE == "SimulatedData":
        d = upload_data("../Simulation/FSN_Data.db", limit=300)
        d = prepare_data(d)

#     Save visualisation of current FSN
    plotFSN(d, edge_labels=False, node_labels=False, title="MarcelExperiment")
