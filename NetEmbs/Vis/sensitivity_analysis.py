# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
sensitivity_analysis.py
Created by lex at 2019-07-12.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

full_names_map = {"V-M": "V-Measure", "FMI": "Fowlkes-Mallows index", "AMI": "Adjusted Mutual Information",
                  "ARI": "Adjusted Rand index"}


def add_average(df, on="V-M", postfix=["_N_global", "_N-1_global", "_N_local", "_N-1_local"], inplace=False):
    """
    Helper function to add a column with average values over the given metric
    :param df: input DataFrame
    :param on: Metric name
    :param postfix: List of postfixes to be used for calculation the average
    :param inplace: If False, return a copy of original DF with added column
    :return: None if inplace=True or DataFrame if inplace=False
    """
    avg_titles = [on + it for it in postfix]
    if inplace:
        df["AVG_" + on] = df.apply(lambda row: np.mean([row[title] for title in avg_titles]), axis=1)
        return
    else:
        df_local = df.copy()
        df_local["AVG_" + on] = df_local.apply(lambda row: np.mean([row[title] for title in avg_titles]), axis=1)
        return df_local


def get_best_combinations(df, on="V-M", postfix=["_N_global", "_N-1_global", "_N_local", "_N-1_local"]):
    df_avg = add_average(df, on=on, postfix=postfix)
    return df_avg.groupby(["Strategy", "Pressure", "Walks per node", "Window size", "Embedding size", "Train steps"]) \
        [["AVG_" + on] + [on + it for it in postfix]].mean() \
        .sort_values("AVG_" + on, ascending=False)


def get_required_metric(df, on="Pressure", metric="V-M"):
    df = add_average(df, on=metric)
    postfix = ["AVG_" + metric] + [metric + it for it in ["_N_global", "_N-1_global", "_N_local", "_N-1_local"]]
    rename_map = dict(zip(postfix, ["Final score", "N clusters, global", "N-1 clusters, global", "N clusters, local",
                                    "N-1 clusters, local"]))
    nominal = {"Walks per node": 30, "Window size": 2, "Pressure": 10, "Embedding size": 8, "Train steps": 50000}
    group_key = list(nominal.keys())
    group_key.remove(on)
    to_plot = df \
        .groupby(group_key).get_group(tuple([nominal[it] for it in group_key]))[[on] + postfix] \
        .rename(rename_map, axis=1)
    return to_plot


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
