# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
helpers.py
Created by lex at 2019-05-02.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def set_font(s=16, reset=False):
    if reset:
        plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["figure.figsize"] = [20, 10]
    #     plt.rcParams['font.family'] = 'serif'
    #     plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rc('font', size=s)  # controls default text sizes
    plt.rc('axes', titlesize=s)  # fontsize of the axes title
    plt.rc('axes', labelsize=s)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=s - 2)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=s - 2)  # fontsize of the tick labels
    plt.rc('legend', fontsize=s)  # legend fontsize
    plt.rc('figure', titlesize=s)  # fontsize of the figure title
