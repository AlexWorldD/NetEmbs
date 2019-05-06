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

def bData():
    df = pd.read_csv("../data/b_processes2.csv", ";")
    df.date = pd.to_datetime(df.date)
    # only a single year
    print("size")
    print(df.shape)
    df = df.loc[df.date > '2015-01-01', :]
    print("single year")
    print(df.shape)
    df.amount = df.amount.apply(lambda x: x.replace(",", "."))
    df.amount = df.amount.astype(float)
    # I need these columns for further work
    df = df[["transactionID", "accountID", "BR", "amount", "type"]]
    return df


if __name__ == '__main__':
    print("Welcome to NetEmbs application!")
    MAIN_LOGGER = log_me()
    MAIN_LOGGER.info("Started..")
    if MODE == "SimulatedData":
        d = upload_data("../Simulation/FSN_Data.db", limit=1000)
        d = prepare_data(d)

    if MODE == "RealData":
        # //////// UPLOAD your data HERE \\\\\\\\\\
        d = bData()
        # //////// END  \\\\\\\\\\
        d = rename_columns(d,
                           names={"transactionID": "ID", "accountID": "FA_Name", "BR": "GroundTruth",
                                  "amount": "Value"})
        d = prepare_dataMarcel(d)
    # Now we should have good and clean dataset
    # let's check it
    countDirtyData(d, ["Debit", "Credit"])
    # Save visualisation of current FSN
    plotFSN(d, edge_labels=False, node_labels=False, title="Marcel/FSN_Vis")

    # ///////// Getting embeddings \\\\\\\\\\\\
    embds = get_embs_TF(d, embed_size=6, walks_per_node=50, num_steps=10000)
    # //////// Merge with GroundTruth \\\\\\\\\
    if MODE == "SimulatedData":
        d = add_ground_truth(embds)
    if MODE == "RealData":
        d = embds.merge(d.groupby("ID", as_index=False).agg({"GroundTruth": "first"}), on="ID")
    d.to_pickle("tmp_dataMarcel.pkl")
    #     ////////// Clustering in embedding space \\\\\\\
    cl_labs = cl_Agglomerative(d, 9)
    print(cl_labs.head(3))
    #     ////////// Plotting tSNE graphs with ground truth vs. labeled \\\\\\\
    plot_tSNE(cl_labs, legend_title="GroundTruth", title="Marcel/GroundTruth")
    plot_tSNE(cl_labs, legend_title="label", title="Marcel/AgglomerativeCl")
