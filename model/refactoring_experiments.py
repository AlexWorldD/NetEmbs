# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
refactoring_experiments.py
Created by lex at 2019-07-04.
"""
from NetEmbs import *
from NetEmbs.Logs.custom_logger import log_me
from NetEmbs.utils import *
import logging

DB_PATH = "../Simulation/FSN_Data.db"

if __name__ == '__main__':
    # Creating current working place for storing intermediate cache and final images
    CONFIG.WORK_FOLDER = ("../UvA/SensitivityAnalysis/" + path_postfix_samplings, path_postfix_tf)
    print(CONFIG.WORK_FOLDER)
    create_working_folder()
    MAIN_LOGGER = log_me()
    MAIN_LOGGER.info("Started..")
    print("Welcome to refactoring experiments!")
    d = upload_data(DB_PATH, limit=None)
    d = prepare_data(d)

    # let's check it
    countDirtyData(d, ["Debit", "Credit"])
    # Save visualisation of current FSN
    # plotFSN(d, edge_labels=False, node_labels=False, title="Marcel/FSN_Vis")
    # ----- SET required parameters in CONFIG file -------
    print("Current config parameters: \n Embedding size: ", EMBD_SIZE, "\n Walks per node: ", WALKS_PER_NODE,
          "\n Steps in TF model: ", STEPS)
    # ///////// Getting embeddings \\\\\\\\\\\\
    try:
        embds = get_embs_TF(d, embed_size=EMBD_SIZE, walks_per_node=WALKS_PER_NODE, num_steps=STEPS,
                            step_version=STEP_VERSION,
                            use_cached_skip_grams=True, use_prev_embs=False, vis_progress=False, groundTruthDF=None)
    except Exception as e:
        if LOG:
            local_logger = logging.getLogger("NetEmbs.MarcelExperiments")
            local_logger.error("We've got an error in get_embs_TF function... ", exc_info=True)
        raise e

    # //////// Add X/Y for plotting and Merge with GroundTruth \\\\\\\\\
    d = add_ground_truth(dim_reduction(embds))
    d.to_pickle(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + "cache/Embeddings.pkl")
    print("Use the following command to see the Tensorboard with all collected stats during last running: \n")
    print("tensorboard --logdir=model/" + CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1])
    #     ////////// Plotting tSNE graphs with ground truth vs. labeled \\\\\\\
    plot_tSNE(d, legend_title="GroundTruth", folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1],
              title="GroundTruth", context="paper_half")
    plot_tSNE(d, legend_title="GroundTruth", folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1],
              title="GroundTruth", context="paper_full")
    print("Plotted the GroundTruth graph!")
