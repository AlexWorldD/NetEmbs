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
from NetEmbs import CONFIG
import numpy as np


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
    if not os.path.exists(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1]):
        os.makedirs(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1], exist_ok=True)
    if not os.path.exists(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + "img/"):
        os.makedirs(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + "img/", exist_ok=True)
    if not os.path.exists(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + "cache/"):
        os.makedirs(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + "cache/", exist_ok=True)
    print("Working directory is ", "model/" + CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1])


def analysisData(db):
    # Creating current working place for storing intermediate cache and final images
    CONFIG.WORK_FOLDER = (db + path_postfix_samplings, path_postfix_tf)
    print(CONFIG.WORK_FOLDER)
    create_working_folder()
    set_font(20)
    print("Welcome to NetEmbs application!")
    MAIN_LOGGER = log_me()
    MAIN_LOGGER.info("Started..")
    if MODE == "SimulatedData":
        d = upload_data(db + "/FSN_Data.db", limit=1000)
        d = prepare_data(d)

    if MODE == "RealData":
        # //////// TODO UPLOAD your data HERE \\\\\\\\\\
        d = bData()
        # //////// END  \\\\\\\\\\
        d = rename_columns(d,
                           names={"transactionID": "ID", "accountID": "FA_Name", "BR": "GroundTruth",
                                  "amount": "Value"})
        # TODO pay attention for the split argument below!
        if "Value" in list(d):
            need_split = True
        else:
            need_split = False
        d = prepare_dataMarcel(d, split=need_split)
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
    d.to_pickle(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + "cache/Embeddings.pkl")
    print("Use the following command to see the Tensorboard with all collected stats during last running: \n")
    print("tensorboard --logdir=model/" + CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1])
    #     ////////// Clustering in embedding space \\\\\\\
    cl_labs = cl_Agglomerative(d, 9)
    print(cl_labs.head(3))
    #     ////////// Plotting tSNE graphs with ground truth vs. labeled \\\\\\\
    plot_tSNE(cl_labs, legend_title="GroundTruth", folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1],
              title="GroundTruth")
    print("Plotted the GroundTruth graph!")
    plot_tSNE(cl_labs, legend_title="label", folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1],
              title="AgglomerativeCl")
    print("Plotted the Clustered graph!")


if __name__ == '__main__':
    for db in ["A", "B"]:
        analysisData(db)

    # Embeddings after small training, 10k for instance
    # embs = pd.read_pickle("model/<YOUR FOLDER TO 10K TRAIN STEPS>/cache/Embeddings.pkl")
    # plotVectors(groupVectors(embs, how="median", subset=["BR4", "BR4c", "BR6", "BR2", "BR3.1", "BR5"]), title="MarcelTest_BadVectors", folder=WORK_FOLDER[0] + WORK_FOLDER[1])
    # # Embeddings after small training, 200k for instance
    # embs = pd.read_pickle("model/<YOUR FOLDER TO 200K TRAIN STEPS>/cache/Embeddings.pkl")
    # plotVectors(groupVectors(embs, how="median", subset=["BR4", "BR4c", "BR6", "BR2", "BR3.1", "BR5"]),
    #             title="MarcelTest_GoodVectors")
