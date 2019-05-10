# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
MarcelExperiments.py
Created by lex at 2019-05-05.
"""
from NetEmbs import *
from NetEmbs.Logs.custom_logger import log_me
import pandas as pd
import os
import numpy as np

# TODO here you can select with what kind of data you are going to work: Simulated or Real one


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


def create_working_folder():
    # Create working folder for current execution
    if not os.path.exists(WORK_FOLDER[0]+WORK_FOLDER[1]):
        os.makedirs(WORK_FOLDER[0]+WORK_FOLDER[1], exist_ok=True)
    if not os.path.exists(WORK_FOLDER[0]+WORK_FOLDER[1] + "img/"):
        os.makedirs(WORK_FOLDER[0]+WORK_FOLDER[1] + "img/", exist_ok=True)
    if not os.path.exists(WORK_FOLDER[0]+WORK_FOLDER[1] + "cache/"):
        os.makedirs(WORK_FOLDER[0]+WORK_FOLDER[1] + "cache/", exist_ok=True)


if __name__ == '__main__':
    # Creating current working place for storing intermediate cache and final images
    create_working_folder()
    print("Welcome to NetEmbs application!")
    MAIN_LOGGER = log_me()
    MAIN_LOGGER.info("Started..")
    if MODE == "SimulatedData":
        d = upload_data("../Simulation/FSN_Data.db", limit=None)
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
    # plotFSN(d, edge_labels=False, node_labels=False, title="Marcel/FSN_Vis")
    # ----- SET required parameters in CONFIG file -------
    print("Current config parameters: \n Embedding size: ", EMBD_SIZE, "\n Walks per node: ", WALKS_PER_NODE,
          "\n Steps in TF model: ", STEPS)
    # ///////// Getting embeddings \\\\\\\\\\\\
    embds = get_embs_TF(d, embed_size=EMBD_SIZE, walks_per_node=WALKS_PER_NODE, num_steps=STEPS,
                        use_cached_skip_grams=True, use_prev_embs=False, vis_progress=False, groundTruthDF=None)
    # //////// Merge with GroundTruth \\\\\\\\\
    if MODE == "SimulatedData":
        d = add_ground_truth(embds)
    if MODE == "RealData":
        d = embds.merge(d.groupby("ID", as_index=False).agg({"GroundTruth": "first"}), on="ID")
    d.to_pickle(WORK_FOLDER[0]+WORK_FOLDER[1] + "cache/Embeddings.pkl")
    #     ////////// Clustering in embedding space \\\\\\\
    cl_labs = cl_Agglomerative(d, 9)
    print(cl_labs.head(3))
    #     ////////// Plotting tSNE graphs with ground truth vs. labeled \\\\\\\
    plot_tSNE(cl_labs, legend_title="GroundTruth", folder=WORK_FOLDER[0]+WORK_FOLDER[1], title="GroundTruth")
    print("Plotted the GroundTruth graph!")
    plot_tSNE(cl_labs, legend_title="label", folder=WORK_FOLDER[0]+WORK_FOLDER[1], title="AgglomerativeCl")
    print("Plotted the Clustered graph!")
