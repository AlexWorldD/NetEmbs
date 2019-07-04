# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
refactoring_experiments.py
Created by lex at 2019-07-04.
"""
from NetEmbs import *
from NetEmbs.Logs.custom_logger import log_me
from NetEmbs.utils import *

DB_PATH = "../Simulation/FSN_Data.db"

if __name__ == '__main__':
    # Creating current working place for storing intermediate cache and final images
    CONFIG.WORK_FOLDER = ("../RefactoringExperiments" + path_postfix_samplings, path_postfix_tf)
    print(CONFIG.WORK_FOLDER)
    create_working_folder()
    MAIN_LOGGER = log_me()
    MAIN_LOGGER.info("Started..")
    print("Welcome to refactoring experiments!")
    d = upload_data(DB_PATH, limit=None)
    d = prepare_data(d)
