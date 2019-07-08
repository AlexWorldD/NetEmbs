# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
End2End.py
Created by lex at 2019-07-08.
"""
from NetEmbs import *
from NetEmbs import CONFIG
from NetEmbs.utils import *
import logging
import pandas as pd
import pickle

# TODO Marcel, replace here ROOT_FOLDER to folder, where you would like to store all tmps and final Results
# DB_NAME = "FSN_Data.db"
# CONFIG.ROOT_FOLDER = "../UvA/LargeDataset/"
DB_NAME = "FSN_Data_5k.db"
CONFIG.ROOT_FOLDER = "../UvA/LargeDataset/"

DB_PATH = "../Simulation/" + DB_NAME
RESULT_FILE = "Results.xlsx"
LIMIT = None

#  ---------- CONFIG Setting HERE ------------
# .1 Sampling parameters
CONFIG.STRATEGY = "MetaDiff"
CONFIG.PRESSURE = 30
CONFIG.WINDOW_SIZE = 2
CONFIG.WALKS_PER_NODE = 30
CONFIG.WALKS_LENGTH = 10
# .2 TF parameters
CONFIG.STEPS = 50000
CONFIG.EMBD_SIZE = 32
CONFIG.LOSS_FUNCTION = "NegativeSampling"  # or "NCE"
CONFIG.BATCH_SIZE = 256
CONFIG.NEGATIVE_SAMPLES = 512
# ---------------------------------------------

N_CL = 11

if __name__ == '__main__':
    create_folder(CONFIG.ROOT_FOLDER)

    # 0. Loggers adding
    log_me(name=CONFIG.MAIN_LOGGER, folder=CONFIG.ROOT_FOLDER, file_name="GlobalLogs")
    logging.getLogger(CONFIG.MAIN_LOGGER).info("Started..")
    # 0.1 Add DataFrame to store the obtain results
    try:
        # Open file with already existing results
        res = pd.read_excel(CONFIG.ROOT_FOLDER + RESULT_FILE, index_col=0)
    except FileNotFoundError as e:
        # If could not find that file, create new empty one
        res = pd.DataFrame(
            columns=['ExperimentNum', 'Strategy', 'Pressure', 'WalkPerNode', 'WalkLength', 'WindowSize', 'EMBD size',
                     'Train steps', 'Batch size', 'Adjusted Rand index', 'Adjusted Mutual Information', 'V-measure',
                     'Fowlkes-Mallows index', 'Sampling time', 'TF time'])
        res.to_excel(CONFIG.ROOT_FOLDER + RESULT_FILE)
    print("Welcome to refactoring experiments!")
    if CONFIG.MODE == "SimulatedData":
        # 1. Upload JournaEntries into memory
        d = upload_data(DB_PATH, limit=LIMIT, logger_name=CONFIG.MAIN_LOGGER)
        journal_truth = upload_JournalEntriesTruth(DB_PATH)[["ID", "GroundTruth", "Time"]]
        # 2. Data pre-processing
        d = prepare_data(d, logger_name=CONFIG.MAIN_LOGGER)

    if CONFIG.MODE == "RealData":
        # //////// TODO UPLOAD your data HERE \\\\\\\\\\
        # d = bData()
        # //////// END  \\\\\\\\\\
        d = rename_columns(d, names={"transactionID": "ID", "accountID": "FA_Name", "BR": "GroundTruth",
                                     "amount": "Value"})
        # TODO pay attention for the split argument below!
        if "Value" in list(d):
            need_split = True
        else:
            need_split = False
        d = prepare_dataMarcel(d, split=need_split, logger_name=CONFIG.MAIN_LOGGER)
        journal_truth = d.groupby("ID", as_index=False).agg({"GroundTruth": "first"})
    # let's check it
    countDirtyData(d, ["Debit", "Credit"])
    # 3. Create Financial Statement Network object
    CONFIG.GLOBAL_FSN = FSN()
    CONFIG.GLOBAL_FSN.build(d, left_title="FA_Name")
    print("FSN sucessfully constructed: \n", CONFIG.GLOBAL_FSN.info())
    logging.getLogger(CONFIG.MAIN_LOGGER).info(f"FSN successfully constructed: \n, {str(CONFIG.GLOBAL_FSN.info())}")
    # 4. Update CONFIG file w.r.t. the new arguments if applicable
    try:
        updateCONFIG()
    except TypeError as e:
        logging.getLogger(CONFIG.MAIN_LOGGER).critical(e)
        raise TypeError("Critical error during CONFIG update. Stop execution!")
    except IOError as e:
        logging.getLogger(CONFIG.MAIN_LOGGER).critical(e)
        raise IOError("Critical error during CONFIG update. Stop execution!")
    cur_params = {"ExperimentNum": CONFIG.EXPERIMENT, "Strategy": CONFIG.STRATEGY,
                  "Pressure": CONFIG.PRESSURE,
                  "WalkPerNode": CONFIG.WALKS_PER_NODE,
                  "WalkLength": CONFIG.WALKS_LENGTH, "WindowSize": CONFIG.WINDOW_SIZE,
                  "EMBD size": CONFIG.EMBD_SIZE,
                  "Train steps": CONFIG.STEPS, "Batch size": CONFIG.BATCH_SIZE}
    print("Loading Embeddings from cache... wait...")
    try:
        with open(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2] + "Embeddings.pkl",
                  "rb") as file:
            embeddings = pickle.load(file)
            run_times = {"Sampling time": 0.0, "TF time": 0.0}
    except FileNotFoundError:
        print("File not found... Recalculate \n")
        print("Sampling sequences... wait...")
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

    #  8.  ////////// Clustering in embedding space \\\\\\\
    cl_labs = cl_Agglomerative(embeddings, N_CL)
    # 8.1 Plot t-SNE visualisation
    plot_tSNE(cl_labs, "label", folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2],
              title="Predicted label",
              context="paper_full")
    plot_tSNE(cl_labs, "GroundTruth",
              folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2],
              title="Ground Truth",
              context="paper_full")
    print("Plotted required graphs!")
    # 8.2 ////////// Evaluate clustering quality \\\\\\\
    all_metrics = evaluate_all(cl_labs)
    # 9. Construct one row with given parameters and obtained results
    cur_params.update(all_metrics)
    cur_params.update(run_times)
    # Upload previous Results file
    res = pd.read_excel(CONFIG.ROOT_FOLDER + RESULT_FILE, index_col=0)
    # Append new result to DataFrame and save as Excel file
    res = res.append(cur_params, ignore_index=True)
    res.to_excel(CONFIG.ROOT_FOLDER + RESULT_FILE)
    print("--------------------------DONE--------------------------")
