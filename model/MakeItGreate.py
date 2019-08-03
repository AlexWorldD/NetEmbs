# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
MakeItGreate.py
Created by lex at 2019-07-29.
"""
from NetEmbs import *
from NetEmbs import CONFIG
from NetEmbs.utils import *
from NetEmbs.SkipGram.construct_skip_grams import get_SkipGrams
from NetEmbs.SkipGram.tf_model.get_embeddings import get_embeddings
import logging

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


def refactoring():
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
        d = data_preprocessing(d)
        journal_truth = d.groupby("ID", as_index=False).agg({"GroundTruth": "first"})
    # let's check it
    count_bad_values(d, ["Debit", "Credit"])
    # 3. Create Financial Statement Network object
    CONFIG.GLOBAL_FSN = FSN()
    CONFIG.GLOBAL_FSN.build(d, left_title="FA_Name")
    print("FSN sucessfully constructed: \n", CONFIG.GLOBAL_FSN.info())
    # 4.  ///////// Getting Skip-Grams \\\\\\\\\\\\
    skip_grams, tr = get_SkipGrams(CONFIG.GLOBAL_FSN, use_cache=True)
    # 5.  ///////// Getting embeddings \\\\\\\\\\\\
    embeddings = get_embeddings(skip_grams, tr)
    # 6. //////// Merge with GroundTruth \\\\\\\\\
    embeddings = embeddings.merge(journal_truth, on="ID")
    #  7.  ////////// Clustering in embedding space \\\\\\\
    cl_labs = cl_Agglomerative(embeddings, N_CL)
    # 7.1 Plot t-SNE visualisation
    draw.embeddings_2D(cl_labs, "label",
                    folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2],
                    title="Predicted label_N",
                    context="talk_full")
    draw.embeddings_2D(cl_labs, "GroundTruth",
                    folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2],
                    title="Ground Truth",
                    context="talk_full")
    print("Plotted required graphs!")


if __name__ == '__main__':
    refactoring()
