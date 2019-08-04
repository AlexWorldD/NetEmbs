# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
SensitivityAnalysis.py
Created by lex at 2019-07-04.
"""
from NetEmbs import *
from NetEmbs import CONFIG
from NetEmbs.utils import *
import logging
import pandas as pd
import numpy as np

CONFIG.MODE = "SimulatedData"
CONFIG.ROOT_FOLDER = "../UvA/Refactoring/"

DB_NAME = "FSN_Data_5k.db"
DB_PATH = "../Simulation/" + DB_NAME
RESULT_FILE = "Results.xlsx"

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

# Mapper from finer GroundTruth to more rough
# E.g. in my case I replace 'Sales 21 btw'/'Sales 6 btw' -> 'Sales'
map_gt = {'Sales 21 btw': 'Sales',
          'Sales 6 btw': 'Sales'}

# -----------
# myGRID_big = {"Strategy": ["MetaDiff"],
#               "Pressure": [1, 10, 30],
#               "Walks_Per_Node": [10, 30, 50],
#               "Window_Size": [1, 2, 3],
#               "Steps": [50000, 100000],
#               "Embd_Size": [16, 32, 48]}


myGRID = {"Strategy": ["MetaUniform", "OriginalRandomWalk", "DefUniform", "MetaDiff"],
          "Window_Size": [2],
          "Pressure": [10],
          "Walks_Per_Node": [30],
          "Embd_Size": [8],
          "Steps": [50000]}

# The number of experiments with the same settings for sampling
SAMPLING_EXPS = 1
# The number of experiments with the same settings for SkipGram model
TF_EXPS = 1
# ~Number of clusters
N_CL = 11

if __name__ == '__main__':
    create_folder(CONFIG.ROOT_FOLDER)
    # 0. Loggers adding
    log_me(name=CONFIG.MAIN_LOGGER, folder=CONFIG.ROOT_FOLDER, file_name="GlobalLogs")
    logging.getLogger(CONFIG.MAIN_LOGGER).info("Started..")

    print("Welcome to Hyperparameters tuning tool!")
    if CONFIG.MODE == "SimulatedData":
        # 1. Upload JournaEntries into memory
        d = upload_data(DB_PATH, limit=LIMIT)
        journal_truth = upload_journal_entries(DB_PATH)[["ID", "GroundTruth", "Time"]]
        # 2. Data pre-processing
        d = data_preprocessing(d)

    if CONFIG.MODE == "RealData":
        # //////// TODO UPLOAD your data HERE \\\\\\\\\\
        # d =
        # //////// END  \\\\\\\\\\
        d = d.rename(index=str, columns={"transactionID": "ID", "accountID": "FA_Name", "BR": "GroundTruth",
                                         "amount": "Value"})
        d = data_preprocessing(d)
        journal_truth = d.groupby("ID", as_index=False).agg({"GroundTruth": "first"})
    # Let's check that data is clean
    count_bad_values(d, ["Debit", "Credit"])
    # 3. Create Financial Statement Network object
    CONFIG.GLOBAL_FSN = FSN()
    CONFIG.GLOBAL_FSN.build(d, left_title="FA_Name")
    print("FSN sucessfully constructed: \n", CONFIG.GLOBAL_FSN.info())
    logging.getLogger(CONFIG.MAIN_LOGGER).info(f"FSN successfully constructed: \n, {str(CONFIG.GLOBAL_FSN.info())}")
    for cur_parameters in get_GRID(myGRID):
        print(f'---------------------------- Current parameters {str(cur_parameters)} ----------------------------')
        logging.getLogger(CONFIG.MAIN_LOGGER).info(
            f'---------------------------- Current parameters {str(cur_parameters)} ----------------------------')
        # Update CONFIG file according to the given arguments
        for key, value in cur_parameters.items():
            setattr(CONFIG, key, value)
        for sampling_exp in range(SAMPLING_EXPS):
            for tf_exp in range(TF_EXPS):
                print(f'-------------- Experiment {(sampling_exp, tf_exp)} --------------')
                logging.getLogger(CONFIG.MAIN_LOGGER).info(
                    f'-------------- Experiment {(sampling_exp, tf_exp)} --------------')
                CONFIG.EXPERIMENT = (sampling_exp, tf_exp)
                # 4. Update CONFIG file w.r.t. the new arguments if applicable
                try:
                    updateCONFIG_4experiments()
                except TypeError as e:
                    logging.getLogger(CONFIG.MAIN_LOGGER).critical(e)
                    raise TypeError("Critical error during CONFIG update. Stop execution!")
                except IOError as e:
                    logging.getLogger(CONFIG.MAIN_LOGGER).critical(e)
                    raise IOError("Critical error during CONFIG update. Stop execution!")
                cur_row = {"ExperimentNum": str(CONFIG.EXPERIMENT), "Strategy": CONFIG.STRATEGY,
                           "Pressure": CONFIG.PRESSURE,
                           "Walks per node": CONFIG.WALKS_PER_NODE,
                           "Walk length": CONFIG.WALKS_LENGTH, "Window size": CONFIG.WINDOW_SIZE,
                           "Embedding size": CONFIG.EMBD_SIZE,
                           "Train steps": CONFIG.STEPS, "Batch size": CONFIG.BATCH_SIZE}
                print("Loading Embeddings from cache... wait...")

                # 4.  ///////// Getting Skip-Grams \\\\\\\\\\\\
                skip_grams, tr = get_SkipGrams(CONFIG.GLOBAL_FSN, use_cache=True, strategy=CONFIG.STRATEGY)
                # 5.  ///////// Getting embeddings \\\\\\\\\\\\
                embeddings = get_embeddings(skip_grams, tr, use_cache=True, use_dim_reduction=True)
                # 6. //////// Merge with GroundTruth \\\\\\\\\
                embeddings = embeddings.merge(journal_truth, on="ID")

                #  7.  ////////// Clustering in embedding space, for N-1 number of cluster, expected output: all sales into one group \\\\\\\
                # Number of collapsed clusters
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
                # 7.1 ////////// Evaluate clustering quality \\\\\\\
                # Global for N-P scale
                cur_eval_results = evaluate_all(cl_labs[~cl_labs.GroundTruth.isin(list(map_gt.keys()))],
                                                column_true=N_P_GroundTruth,
                                                postfix="_N-" + str(P) + "_global")
                cur_row.update(cur_eval_results)
                cur_score_to_show.append(cur_eval_results["V-M_N-" + str(P) + "_global"])
                # Local for N-P scale
                cur_eval_results = evaluate_all(cl_labs[cl_labs.GroundTruth.isin(list(map_gt.keys()))],
                                                column_true=N_P_GroundTruth,
                                                postfix="_N-" + str(P) + "_local")
                cur_row.update(cur_eval_results)
                cur_score_to_show.append(cur_eval_results["V-M_N-" + str(P) + "_local"])
                # 7.2 Plot t-SNE visualisation
                draw.embeddings_2D(cl_labs, legend_title=N_P_GroundTruth,
                                   folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2],
                                   title="GroundTruth_N-" + str(P),
                                   context="talk_full", save=True)
                draw.embeddings_2D(cl_labs, legend_title="label",
                                   folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2],
                                   title="Predicted label_N-" + str(P),
                                   context="talk_full", save=True, set_score=np.mean(cur_score_to_show))
                # 7.5  ////////// Clustering in embedding space with N clusters \\\\\\\
                cl_labs = cl_Agglomerative(embeddings, N_CL)
                cur_score_to_show = list()
                # 7.6 ////////// Evaluate clustering quality \\\\\\\
                # Global for N scale
                cur_eval_results = evaluate_all(cl_labs[~cl_labs.GroundTruth.isin(list(map_gt.keys()))],
                                                column_true="GroundTruth",
                                                postfix="_N_global")
                cur_row.update(cur_eval_results)
                cur_score_to_show.append(cur_eval_results["V-M_N_global"])
                # Local for N scale
                cur_eval_results = evaluate_all(cl_labs[cl_labs.GroundTruth.isin(list(map_gt.keys()))],
                                                column_true="GroundTruth",
                                                postfix="_N_local")
                cur_row.update(cur_eval_results)
                cur_score_to_show.append(cur_eval_results["V-M_N_local"])
                # 8.1 Plot t-SNE visualisation
                draw.embeddings_2D(cl_labs, legend_title="GroundTruth",
                                   folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2],
                                   title="Ground Truth",
                                   context="talk_full", save=True)
                draw.embeddings_2D(cl_labs, legend_title="label",
                                   folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2],
                                   title="Predicted label_N",
                                   context="talk_full", save=True, set_score=np.mean(cur_score_to_show))
                print("Plotted required graphs!")
                try:
                    # Upload previous Results file
                    res = pd.read_excel(CONFIG.ROOT_FOLDER + RESULT_FILE, index_col=0)
                    # Append new result to DataFrame and save as Excel file
                    res = res.append(cur_row, ignore_index=True)
                    res.to_excel(CONFIG.ROOT_FOLDER + RESULT_FILE)
                except FileNotFoundError:
                    # Create new Results file if not exist
                    res = pd.DataFrame(cur_row, index=[0])
                    res.to_excel(CONFIG.ROOT_FOLDER + RESULT_FILE)
