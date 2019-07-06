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
import logging


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
        d = upload_data(db + "/FSN_Data.db", limit=None)
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
    try:
        embds = get_embs_TF(d, embed_size=EMBD_SIZE, walks_per_node=WALKS_PER_NODE, num_steps=STEPS, step_version=STEP_VERSION,
                        use_cached_skip_grams=True, use_prev_embs=False, vis_progress=False, groundTruthDF=None)
    except Exception as e:
        if LOG:
            local_logger = logging.getLogger("NetEmbs.MarcelExperiments")
            local_logger.error("We've got an error in get_embs_TF function... ", exc_info=True)
        raise e

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


def pairWiseSim(N, mode="DB"):
    def getVectorized(row):
        signatureL = dict(sorted(
            list(
                zip(row["FA_Name"][row["Credit"] > 0.0].values, row["Credit"][row["Credit"] > 0.0].values)),
            key=lambda x: x[0]))
        signatureR = dict(sorted(
            list(zip(row["FA_Name"][row["Debit"] > 0.0].values, row["Debit"][row["Debit"] > 0.0].values)),
            key=lambda x: x[0]))

        cur_left = ALL_FAs.copy()
        cur_right = ALL_FAs.copy()
        cur_left.update(signatureL)
        cur_right.update(signatureR)
        return pd.Series(
            {"ID": row["ID"].values[0], "OneHots": np.array(list(cur_left.values()) + list(cur_right.values()))})

    def getVectorizedDF(df):
        res = df.groupby("ID", as_index=False).apply(getVectorized)
        return res

    vect_mean = np.vectorize(lambda x: np.mean(x))
    from sklearn import preprocessing
    def getPairWise(df):
        # For exactly the same BPs one gets 0.0, hence, in terms of similarity we should substruct it from 1.0
        arr = preprocessing.normalize(1.0-vect_mean(np.power(np.subtract.outer(*[df.OneHots] * 2).T, 2)), axis=1)
        return pd.concat((df['ID'], pd.DataFrame(arr, columns=df['ID'])), axis=1)

    if mode == "DB":
        d = upload_data("../Simulation/FSN_Data.db", limit=N)
        d = prepare_data(d)
    else:
        from NetEmbs.GenerateData.complex_df import sales_collections
        d = sales_collections(N, noise=[])
        d = prepare_data(d, split=False)

    ALL_FAs = dict(zip(sorted(d.FA_Name.unique()), [0.0] * d.FA_Name.nunique()))
    vect_d = getVectorizedDF(d)

    res = getPairWise(vect_d)
    res.set_index("ID", inplace=True)
    import matplotlib.pyplot as plt
    import seaborn as sns
    # plot heatmap
    ax = sns.heatmap(res.T, cmap="Blues")

    # turn the axis label
    for item in ax.get_yticklabels():
        item.set_rotation(0)

    for item in ax.get_xticklabels():
        item.set_rotation(90)
    plt.show()
    print(d.shape)


if __name__ == '__main__':
    # pairWiseSim(100, mode="DB")

    for db in ["../Simulation"]:
    # for db in ["A", "B"]:
        analysisData(db)

    # Embeddings after small training, 10k for instance
    # embs = pd.read_pickle("model/<YOUR FOLDER TO 10K TRAIN STEPS>/cache/Embeddings.pkl")
    # plotVectors(groupVectors(embs, how="median", subset=["BR4", "BR4c", "BR6", "BR2", "BR3.1", "BR5"]), title="MarcelTest_BadVectors", folder=WORK_FOLDER[0] + WORK_FOLDER[1])
    # # Embeddings after small training, 200k for instance
    # embs = pd.read_pickle("model/<YOUR FOLDER TO 200K TRAIN STEPS>/cache/Embeddings.pkl")
    # plotVectors(groupVectors(embs, how="median", subset=["BR4", "BR4c", "BR6", "BR2", "BR3.1", "BR5"]),
    #             title="MarcelTest_GoodVectors")
