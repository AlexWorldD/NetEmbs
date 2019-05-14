# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
custom_logger.py
Created by lex at 2019-04-12.
"""
import logging
from NetEmbs import CONFIG


def log_me(name="NetEmbs", file_name="logs.log", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    file_name = CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + file_name
    # create the logging file handler
    with open(file_name, 'w'):
        pass
    fh = logging.FileHandler(file_name)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add handler to logger object
    logger.addHandler(fh)
    return logger
