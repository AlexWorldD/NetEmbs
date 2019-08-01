# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
update_config.py
Created by lex at 2019-07-07.
"""
from NetEmbs import CONFIG
from NetEmbs.utils.IO.folders_creation import create_working_folder
import logging
from NetEmbs.utils.Logs import log_me


def updateCONFIG_4experiments(create_folder=True):
    # Current working folder
    CONFIG.path_postfix_samplings = "ver" + str(CONFIG.STRATEGY) \
                                    + "_dir" + str(CONFIG.DIRECTION) \
                                    + "_walks" + str(CONFIG.WALKS_PER_NODE) \
                                    + "_pressure" + str(CONFIG.PRESSURE) \
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


def set_new_config(**kwargs) -> None:
    """
    Set new parameters to CONFIG file for further execuiton
    """
    changes: int = 0
    for key, value in kwargs.items():
        if value is not None and value != getattr(CONFIG, key.upper()):
            changes += 1
            setattr(CONFIG, key.upper(), value)
    if changes:
        CONFIG.path_postfix_samplings = "ver" + str(CONFIG.STRATEGY) \
                                        + "_dir" + str(CONFIG.DIRECTION) \
                                        + "_walks" + str(CONFIG.WALKS_PER_NODE) \
                                        + "_pressure" + str(CONFIG.PRESSURE) + "/"
        CONFIG.path_postfix_tf = "EMB" + str(CONFIG.EMBD_SIZE) \
                                 + "_batch" + str(CONFIG.BATCH_SIZE) \
                                 + "_NegSamples" + str(CONFIG.NEGATIVE_SAMPLES) \
                                 + "_TFsteps" + str(CONFIG.STEPS)
        CONFIG.path_postfix_win = "windowSize" + str(CONFIG.WINDOW_SIZE) + "/"

        # The following structure of folders: Sampling parameters -> Window parameter -> TF parameters
        CONFIG.WORK_FOLDER = (
            CONFIG.ROOT_FOLDER + CONFIG.path_postfix_samplings, CONFIG.path_postfix_win, CONFIG.path_postfix_tf)
        print("Config file has been updated!")
        local_logger = logging.getLogger(CONFIG.MAIN_LOGGER + ".CONFIG")
        local_logger.info("Config file has been updated!")
        local_logger.info(f"Working folder is \n {str(CONFIG.WORK_FOLDER)}")
        create_working_folder()
        # Local logging added
        log_me(name="NetEmbs", file_name="LocalLogs.log").info("Started..")
