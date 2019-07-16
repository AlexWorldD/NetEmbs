# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
results_processing.py
Created by lex at 2019-07-14.
"""
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
