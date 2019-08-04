# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
test_end2end.py
Created by lex at 2019-08-04.
"""
from NetEmbs import *
from NetEmbs import CONFIG
from NetEmbs.utils import *
from NetEmbs.SkipGram.construct_skip_grams import get_SkipGrams
from NetEmbs.SkipGram.tf_model.get_embeddings import get_embeddings
import logging
import shutil

CONFIG.ROOT_FOLDER = "tests/CompleteRun/"

DB_NAME = "FSN_Data_5k.db"
DB_PATH = "tests/CompleteRun/" + DB_NAME

LIMIT = 1000
#  ---------- CONFIG Setting HERE ------------
# .1 Sampling parameters
CONFIG.STRATEGY = "MetaDiff"
CONFIG.PRESSURE = 10
CONFIG.WINDOW_SIZE = 2
CONFIG.WALKS_PER_NODE = 30
CONFIG.WALKS_LENGTH = 10
# .2 TF parameters
CONFIG.STEPS = 10000

CONFIG.EMBD_SIZE = 8
CONFIG.LOSS_FUNCTION = "NegativeSampling"  # or "NCE"
CONFIG.BATCH_SIZE = 32
CONFIG.NEGATIVE_SAMPLES = 64
# ---------------------------------------------

# -----------
# E.g. in my case I replace 'Sales 21 btw'/'Sales 6 btw' -> 'Sales'
map_gt = {'Sales 21 btw': 'Sales',
          'Sales 6 btw': 'Sales'}

# ~Number of clusters
N_CL = 11


def make_one_run() -> str:
    """
    Test by executing the complete pipile from raw data to the clustering and plotting.
    Returns
    -------
    'SUCCESS' in case of no errors within the run.
    """
    try:
        updateCONFIG_4experiments(create_folder=True)
    except TypeError as e:
        logging.getLogger(CONFIG.MAIN_LOGGER).critical(e)
        raise TypeError("Critical error during CONFIG update. Stop execution!")
    except IOError as e:
        logging.getLogger(CONFIG.MAIN_LOGGER).critical(e)
        raise IOError("Critical error during CONFIG update. Stop execution!")

    # 0. Loggers adding
    log_me(name=CONFIG.MAIN_LOGGER, folder=CONFIG.ROOT_FOLDER, file_name="GlobalLogs")
    logging.getLogger(CONFIG.MAIN_LOGGER).info("Started..")

    # 1. Upload JournaEntries into memory
    d = upload_data(DB_PATH, limit=LIMIT)
    journal_truth = upload_journal_entries(DB_PATH)[["ID", "GroundTruth", "Time"]]
    # 2. Data pre-processing
    d = data_preprocessing(d)
    # let's check it
    count_bad_values(d, ["Debit", "Credit"])
    # 3. Create Financial Statement Network object
    CONFIG.GLOBAL_FSN = FSN()
    CONFIG.GLOBAL_FSN.build(d, left_title="FA_Name")
    # 4.  ///////// Getting Skip-Grams \\\\\\\\\\\\
    skip_grams, tr = get_SkipGrams(CONFIG.GLOBAL_FSN, use_cache=True)
    # 5.  ///////// Getting embeddings \\\\\\\\\\\\
    embeddings = get_embeddings(skip_grams, tr, use_cache=True)
    # 6. //////// Merge with GroundTruth \\\\\\\\\
    embeddings = embeddings.merge(journal_truth, on="ID")
    #  7.  ////////// Clustering in embedding space \\\\\\\
    cl_labs = cl_Agglomerative(embeddings, N_CL)
    # 7.1 Plot t-SNE visualisation
    draw.embeddings_2D(cl_labs, legend_title="label",
                       folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2],
                       title="Predicted label_N",
                       context="talk_full", save=True)

    # 7.2 Evaluate the clustering
    cur_row = {"Strategy": CONFIG.STRATEGY,
               "Pressure": CONFIG.PRESSURE,
               "Walks per node": CONFIG.WALKS_PER_NODE, "Window size": CONFIG.WINDOW_SIZE,
               "Embedding size": CONFIG.EMBD_SIZE, "Train steps": CONFIG.STEPS}
    # Global for N scale
    cur_eval_results = evaluate_all(cl_labs[~cl_labs.GroundTruth.isin(list(map_gt.keys()))],
                                    column_true="GroundTruth",
                                    postfix="_N_global")
    cur_row.update(cur_eval_results)
    # Local for N scale
    cur_eval_results = evaluate_all(cl_labs[cl_labs.GroundTruth.isin(list(map_gt.keys()))],
                                    column_true="GroundTruth",
                                    postfix="_N_local")
    cur_row.update(cur_eval_results)
    return "SUCCESS"


def test_complete_run():
    assert 'SUCCESS' == make_one_run()
    shutil.rmtree(CONFIG.WORK_FOLDER[0])
