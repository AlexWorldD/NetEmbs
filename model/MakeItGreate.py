# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
MakeItGreate.py
Created by lex at 2019-07-29.
"""
from NetEmbs import *
from NetEmbs import CONFIG
from NetEmbs.utils import *
import logging
import pandas as pd
import pickle

DB_NAME = "FSN_Data_5k.db"
CONFIG.ROOT_FOLDER = "../UvA/Refactoring/"

RESULT_FILE = "ResultsRefactoring.xlsx"

DB_PATH = "../Simulation/" + DB_NAME

LIMIT = None

#  ---------- CONFIG Setting HERE ------------
# .1 Sampling parameters
CONFIG.STRATEGY = "MetaDiff"
CONFIG.PRESSURE = 10
CONFIG.WINDOW_SIZE = 2
CONFIG.WALKS_PER_NODE = 30
CONFIG.WALKS_LENGTH = 10
# .2 TF parameters
CONFIG.STEPS = 50000

CONFIG.EMBD_SIZE = 8
CONFIG.LOSS_FUNCTION = "NegativeSampling"  # or "NCE"
CONFIG.BATCH_SIZE = 256
CONFIG.NEGATIVE_SAMPLES = 512
# ---------------------------------------------

# -----------
# E.g. in my case I replace 'Sales 21 btw'/'Sales 6 btw' -> 'Sales'
map_gt = {'Sales 21 btw': 'Sales',
          'Sales 6 btw': 'Sales'}

# ~Number of clusters
N_CL = 11

if __name__ == '__main__':
    create_folder(CONFIG.ROOT_FOLDER)
    try:
        updateCONFIG_4experiments()
    except TypeError as e:
        logging.getLogger(CONFIG.MAIN_LOGGER).critical(e)
        raise TypeError("Critical error during CONFIG update. Stop execution!")
    except IOError as e:
        logging.getLogger(CONFIG.MAIN_LOGGER).critical(e)
        raise IOError("Critical error during CONFIG update. Stop execution!")

    # 0. Loggers adding
    log_me(name=CONFIG.MAIN_LOGGER, folder=CONFIG.ROOT_FOLDER, file_name="GlobalLogs")
    logging.getLogger(CONFIG.MAIN_LOGGER).info("Started..")

    print("Welcome to refactoring experiments!")
    if CONFIG.MODE == "SimulatedData":
        # 1. Upload JournaEntries into memory
        d = upload_data(DB_PATH, limit=LIMIT)
        journal_truth = upload_journal_entries(DB_PATH)[["ID", "GroundTruth", "Time"]]
        # 2. Data pre-processing
        d = data_preprocessing(d)

    if CONFIG.MODE == "RealData":
        # //////// TODO UPLOAD your data HERE \\\\\\\\\\
        # d = bData()
        # //////// END  \\\\\\\\\\
        d = d.rename(index=str, columns={"transactionID": "ID", "accountID": "FA_Name", "BR": "GroundTruth",
                                         "amount": "Value"})
        # TODO pay attention for the split argument below!
        if "Value" in list(d):
            need_split = True
        else:
            need_split = False
        d = data_preprocessing(d)
        journal_truth = d.groupby("ID", as_index=False).agg({"GroundTruth": "first"})
    # let's check it
    count_bad_values(d, ["Debit", "Credit"])
    # 3. Create Financial Statement Network object
    CONFIG.GLOBAL_FSN = FSN()
    CONFIG.GLOBAL_FSN.build(d, left_title="FA_Name")
    print("FSN sucessfully constructed: \n", CONFIG.GLOBAL_FSN.info())
    # 5.  ///////// Getting embeddings \\\\\\\\\\\\
    try:
        embeddings, run_times = get_embs_TF(evaluate_time=True)
    except Exception as e:
        logging.getLogger(CONFIG.MAIN_LOGGER).error("We've got an error in get_embs_TF function... ",
                                                    exc_info=True)
        # 6. //////// Merge with GroundTruth \\\\\\\\\
    embeddings = embeddings.merge(journal_truth, on="ID")

        # 7. Dimensionality reduction for visualisation purposes
    embeddings = dim_reduction(embeddings)
    embeddings.to_pickle(
            CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2] + "Embeddings.pkl")

        #  8.  ////////// Clustering in embedding space, for N-1 number of cluster, expected output: all sales into one group \\\\\\\
    map_gt = {title: map_gt.get(title) for title in embeddings.GroundTruth.unique()}
    map_gt = {key: item for key, item in map_gt.items() if item is not None}
    cur_score_to_show = list()
    P = len(set(map_gt.keys())) - len(set(map_gt.values()))
    print(f"You are going to cluster with {N_CL}-{P}")
    # Column title for new ground truth
    N_P_GroundTruth = "GroundTruthN-" + str(P)
    embeddings[N_P_GroundTruth] = embeddings["GroundTruth"]
    embeddings[N_P_GroundTruth] = embeddings[N_P_GroundTruth].apply(
        lambda x: map_gt.get(x) if map_gt.get(x) is not None else x)
    cl_labs = cl_Agglomerative(embeddings, N_CL - P)
    # Global for N-P scale
    cur_eval_results = evaluate_all(cl_labs[~cl_labs.GroundTruth.isin(list(map_gt.keys()))],
                                    column_true=N_P_GroundTruth,
                                    postfix="_N-" + str(P) + "_global")
    cur_score_to_show.append(cur_eval_results["V-M_N-" + str(P) + "_global"])
    # Local for N-P scale
    cur_eval_results = evaluate_all(cl_labs[cl_labs.GroundTruth.isin(list(map_gt.keys()))],
                                    column_true=N_P_GroundTruth,
                                    postfix="_N-" + str(P) + "_local")
    cur_score_to_show.append(cur_eval_results["V-M_N-" + str(P) + "_local"])
    # 8.1 Plot t-SNE visualisation
    plot_tSNE(cl_labs, N_P_GroundTruth,
              folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2],
              title="GroundTruth_N-" + str(P),
              context="talk_half")
    plot_tSNE(cl_labs, "label",
              folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2],
              title="Predicted label_N-" + str(P),
              context="talk_half", score=cur_score_to_show)
    #  8.  ////////// Clustering in embedding space \\\\\\\
    # TODO Marcel: clustering into N group and again evaluation
    cl_labs = cl_Agglomerative(embeddings, N_CL)
    cur_score_to_show = list()
    # 8.2 ////////// Evaluate clustering quality \\\\\\\
    # Global for N scale
    cur_eval_results = evaluate_all(cl_labs[~cl_labs.GroundTruth.isin(list(map_gt.keys()))],
                                    column_true="GroundTruth",
                                    postfix="_N_global")
    cur_score_to_show.append(cur_eval_results["V-M_N_global"])
    # Local for N scale
    cur_eval_results = evaluate_all(cl_labs[cl_labs.GroundTruth.isin(list(map_gt.keys()))],
                                    column_true="GroundTruth",
                                    postfix="_N_local")
    cur_score_to_show.append(cur_eval_results["V-M_N_local"])
    # 8.1 Plot t-SNE visualisation
    plot_tSNE(cl_labs, "label",
              folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2],
              title="Predicted label_N",
              context="talk_half", score=cur_score_to_show)
    plot_tSNE(cl_labs, "GroundTruth",
              folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2],
              title="Ground Truth",
              context="talk_half")
    print("Plotted required graphs!")