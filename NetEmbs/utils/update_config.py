# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
update_config.py
Created by lex at 2019-07-07.
"""
from NetEmbs import CONFIG
from NetEmbs.utils.IO.create_work_folder import create_working_folder
import logging
from NetEmbs.utils.Logs import log_me


def updateCONFIG(create_folder=True):
    # Current working folder
    CONFIG.path_postfix_samplings = "ver" + str(CONFIG.STRATEGY) \
                                    + "_dir" + str(CONFIG.DIRECTION) \
                                    + "_walks" + str(CONFIG.WALKS_PER_NODE) \
                                    + "_pressure" + str(CONFIG.PRESSURE) \
                                    + "_1hopFraction" + str(CONFIG.HACK) \
                                    + "_" + str(CONFIG.EXPERIMENT[0]) + "/"
    CONFIG.path_postfix_tf = "EMB" + str(CONFIG.EMBD_SIZE) \
                             + "_batch" + str(CONFIG.BATCH_SIZE) \
                             + "_NegSamples" + str(CONFIG.NEGATIVE_SAMPLES) \
                             + "_TFsteps" + str(CONFIG.STEPS) \
                             + "_" + str(CONFIG.EXPERIMENT[1]) + "/"
    CONFIG.path_postfix_win = "windowSize" + str(CONFIG.WINDOW_SIZE) + "/"

    # The following structure of folders: Sampling parameters -> Window parameter -> TF parameters
    CONFIG.WORK_FOLDER = (
        CONFIG.ROOT_FOLDER + CONFIG.path_postfix_samplings, CONFIG.path_postfix_win, CONFIG.path_postfix_tf)
    print("Config file has been updated!")
    if create_folder:
        local_logger = logging.getLogger(CONFIG.MAIN_LOGGER + ".CONFIG")
        local_logger.info("Config file has been updated!")
        local_logger.info(f"Working folder is \n {str(CONFIG.WORK_FOLDER)}")
        create_working_folder()
        # Local logging added
        log_me(name="NetEmbs", file_name="LocalLogs.log").info("Started..")
    return
