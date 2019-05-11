# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
plot_vectors.py
Created by lex at 2019-05-09.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np

def plotVectors(df, title="Vectors"):
    plt.rc('axes', titlesize=18)  # fontsize of the x and y titles
    plt.rc('axes', labelsize=18)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)  # fontsize of the tick labels
    plt.figure(figsize=(20, 8))
    t_ax = sns.heatmap(list(df["Emb"].values), vmin=-1.0, vmax=1.0, cmap=sns.color_palette("RdBu_r", 10))
    frame1 = plt.gca()
    frame1.axes.set_xlabel('')
    frame1.axes.set_ylabel('Business processes')
    frame1.axes.xaxis.set_ticklabels([])
    ns = np.where(df.GroundTruth.values == None)[0][0]
    frame1.axes.yaxis.set_ticklabels(list(df["GroundTruth"].dropna().unique()), rotation='horizontal')
    frame1.axes.yaxis.set_major_locator(ticker.FixedLocator([ns / 2 + it * (ns + 1) for it in range(8)]))
    postfix = "_emb_size" + str(len(df["Emb"].values[0])) + "samples_per_group" + str(ns)
    plt.tight_layout()
    plt.savefig("img/" + title + postfix, dpi=140, pad_inches=0.01)
