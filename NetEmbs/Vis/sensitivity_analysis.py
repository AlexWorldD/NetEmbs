# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
sensitivity_analysis.py
Created by lex at 2019-07-12.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from NetEmbs.utils.results_processing import *

full_names_map = {"V-M": "V-Measure", "FMI": "Fowlkes-Mallows index", "AMI": "Adjusted Mutual Information",
                  "ARI": "Adjusted Rand index"}



def plotSensitivity(df, on="Pressure", metric="V-M", mode="final"):
    df = get_required_metric(df, on=on, metric=metric)
    sns.set_context("paper", font_scale=2.3)
    if mode == "final":
        fig_size = (6.4, 4.8)
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        sns.lineplot(x=df.columns[0], y="Final score", data=df, marker="s", markersize=7, err_style="band",
                     err_kws={"alpha": 0.08}, ax=ax)
    elif mode == "all":
        df = df.set_index(on).stack().reset_index().rename({"level_1": "", 0: "score"}, axis=1)
        fig_size = (12, 8)
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        #         df["alpha"] = np.where(df[""] == "Final score", 1.0, 0.5)
        sns.lineplot(x=df.columns[0], y="score", hue="", alpha=0.25, data=df[df[""] != "Final score"], marker="s",
                     markersize=7, err_style="band", err_kws={"alpha": 0.05}, ax=ax)
        sns.lineplot(x=df.columns[0], y="score", hue="", alpha=1, lw=2, data=df[df[""] == "Final score"], marker="s",
                     markersize=7, err_style="band", err_kws={"alpha": 0.1}, ax=ax)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, markerscale=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel(df.columns[0], labelpad=15)
    ax.set_ylabel(full_names_map[metric], labelpad=15)
    fig.savefig(
        "img/SensitivityAnalysis/" + full_names_map[metric] + "_ErrorBar_for_" + df.columns[0] + "_" + mode + ".png",
        bbox_inches="tight", dpi=140, pad_inches=0.05)
    plt.close(fig)
