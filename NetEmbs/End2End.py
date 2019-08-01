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
import pickle

# TODO Marcel: replace here ROOT_FOLDER to folder, where you would like to store all tmps and final Results
# CONFIG.ROOT_FOLDER = "../UvA/SensitivityAnalysis/"
# DB_NAME = "FSN_Data.db"
DB_NAME = "FSN_Data_5k.db"
CONFIG.ROOT_FOLDER = "../UvA/LargeDataset/"

RESULT_FILE = "ResultsSensitivity.xlsx"

DB_PATH = "../Simulation/" + DB_NAME

LIMIT = None

#  ---------- CONFIG Setting HERE ------------
# .1 Sampling parameters
CONFIG.STRATEGY = "MetaDiff"
CONFIG.PRESSURE = 20
CONFIG.WINDOW_SIZE = 2
CONFIG.WALKS_PER_NODE = 10
CONFIG.WALKS_LENGTH = 10
# .2 TF parameters
CONFIG.STEPS = 50000
# TODO Marcel: According to my experiment, even 8 is fine.
#  Plus that number is fit to formula from Google for embeddings size, see here - https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
CONFIG.EMBD_SIZE = 8
CONFIG.LOSS_FUNCTION = "NegativeSampling"  # or "NCE"
CONFIG.BATCH_SIZE = 256
CONFIG.NEGATIVE_SAMPLES = 512
# ---------------------------------------------

# -----------
# TODO Marcel: specify here the dict map for N-1 GroundTruth.
#  It's stupid to do it in that way, but for a few runs it should be fine.
# E.g. in my case I replace 'Sales 21 btw'/'Sales 6 btw' -> 'Sales'
map_gt = {'Fixed Assets': 'Fixed Assets',
          'Sales 21 btw': 'Sales',
          'Goods delivery': 'Goods delivery',
          'Payroll': 'Payroll',
          'Sales tax': 'Sales tax',
          'Disbursement': 'Disbursement',
          'Payroll Disbursement': 'Payroll Disbursement',
          'Sales 6 btw': 'Sales',
          'Purchase': 'Purchase',
          'Depreciation': 'Depreciation',
          'Collections': 'Collections'}
# TODO Marcel: specify here the GroundTruth values which should be used for local evaluation
#  the ones that from one group in general, but with different rates etc.
#  I guess, it's something like BR4.1, BR4.2 etc. with common first digital?
local_titles = ['Sales 6 btw', 'Sales 21 btw']
# -----------
# myGRID_big = {"Strategy": ["MetaDiff"],
#               "Pressure": [1, 10, 30],
#               "Walks_Per_Node": [10, 30, 50],
#               "Window_Size": [1, 2, 3],
#               "Steps": [50000, 100000],
#               "Embd_Size": [16, 32, 48]}

# TODO Marcel: Try that grid: 90 combinations X 4 experiment, 360 in total. BUT, for Embedding size 16 you should have cache
#  as well as for the majority of sampling parameters, hence, it definitely should be faster that your prev runs :)
#  or if your prev run has already finished, you could repeat it here, get new Excel file with new metrics. It also could be good :)

myGRID = {"Strategy": ["MetaDiff"],
          "Window_Size": [2],
          "Pressure": [10],
          "Walks_Per_Node": [30],
          "Embd_Size": [8],
          "Steps": [100000]}

# The number of experiments with the same settings for sampling
SAMPLING_EXPS = 2
# The number of experiments with the same settings for SkipGram model
TF_EXPS = 2
# TODO Marcel: I guess here you have to set the number of unique values in GroundTruth for your dataset
#  or prior to execution try manually leave only important GroundTruth in dataset...
#  Because if we have quite uncertain method and test it with quite uncertain dataset - it is hard to justify the results.
# ~Number of clusters
N_CL = 11

create_folder(CONFIG.ROOT_FOLDER)
# 0. Loggers adding
log_me(name=CONFIG.MAIN_LOGGER, folder=CONFIG.ROOT_FOLDER, file_name="GlobalLogs")
logging.getLogger(CONFIG.MAIN_LOGGER).info("Started..")

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
            # TODO update CONFIG values and create tmps folders
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
                    raise Exception("We've got an error in get_embs_TF function... ")

                # 6. //////// Merge with GroundTruth \\\\\\\\\
                embeddings = embeddings.merge(journal_truth, on="ID")

                # 7. Dimensionality reduction for visualisation purposes
                embeddings = dim_reduction(embeddings)
                embeddings.to_pickle(
                    CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2] + "Embeddings.pkl")

            #  8.  ////////// Clustering in embedding space, for N-1 number of cluster, expected output: all sales into one group \\\\\\\
            # TODO Marcel: here we cluster into N-1 group
            embeddings["GroundTruthN-1"] = embeddings["GroundTruth"].apply(lambda x: map_gt[x])
            cl_labs = cl_Agglomerative(embeddings, N_CL - 1)
            cur_row.update(
                evaluate_all(cl_labs[~cl_labs.GroundTruth.isin(local_titles)], column_true="GroundTruthN-1",
                             postfix="_N-1_global"))
            cur_row.update(
                evaluate_all(cl_labs[cl_labs.GroundTruth.isin(local_titles)], column_true="GroundTruthN-1",
                             postfix="_N-1_local"))
            # 8.1 Plot t-SNE visualisation
            plot_tSNE(cl_labs, "label",
                      folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2],
                      title="Predicted label_N-1",
                      context="paper_full")
            #  8.  ////////// Clustering in embedding space \\\\\\\
            # TODO Marcel: clustering into N group and again evaluation
            cl_labs = cl_Agglomerative(embeddings, N_CL)
            # 8.1 Plot t-SNE visualisation
            plot_tSNE(cl_labs, "label",
                      folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2],
                      title="Predicted label_N",
                      context="paper_full")
            plot_tSNE(cl_labs, "GroundTruth",
                      folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2],
                      title="Ground Truth",
                      context="paper_full")
            print("Plotted required graphs!")
            # 8.2 ////////// Evaluate clustering quality \\\\\\\
            cur_row.update(
                evaluate_all(cl_labs[~cl_labs.GroundTruth.isin(local_titles)], column_true="GroundTruth",
                             postfix="_N_global"))
            cur_row.update(
                evaluate_all(cl_labs[cl_labs.GroundTruth.isin(local_titles)], column_true="GroundTruth",
                             postfix="_N_local"))
            # 9. Construct one row with given parameters and obtained results
            cur_row.update(run_times)
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


