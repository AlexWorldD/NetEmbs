# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
__init__.py.py
Created by lex at 2019-03-15.
"""
from NetEmbs.Vis.plots import plotFSN, plotHist, plotHeatMap, plot_tSNE, plot_PCA
from NetEmbs.Vis.helpers import set_font, getColors_Markers
from NetEmbs.Vis.plot_vectors import plotVectors
from NetEmbs.Vis.forModelling.CorrelationHeatMap import corrHeatmap_interactive, corrHeatmap_static
from NetEmbs.Vis.sensitivity_analysis import plotSensitivity
