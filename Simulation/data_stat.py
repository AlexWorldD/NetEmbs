# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
data_stat.py
Created by lex at 2019-04-23.
"""

from NetEmbs.DataProcessing import *
from Simulation.CreateDB import *
from Simulation.FSN_Simulation import FSN_Simulation
from NetEmbs.Vis.plots import plotHist

if __name__ == '__main__':
    d = prepare_data(upload_data())
    plotHist(d)
    print("t")
