# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
custom_logger.py
Created by lex at 2019-04-12.
"""
import logging
from NetEmbs import CONFIG
from typing import Optional


def log_me(name: str = "NetEmbs", folder: Optional[str] = None, file_name: Optional[str] = "logs.log",
           level: Optional[int] = logging.INFO) -> logging.Logger:
    """
    Attach logger to specific file and location

    Parameters
    ----------
    name : str, default is 'NetEmbs'
        The logger name.
    folder : str, optional, default is None
        Folder where to store the log file.
    file_name : str, optional, default is 'logs.log'
    level : int, optional, default is INFO (=20)

    Returns
    -------
    Logger with chosen name and chosen location.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if folder is not None:
        file_name = folder + file_name
    else:
        file_name = CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2] + file_name
    # create the logging file handler
    with open(file_name, 'w'):
        pass
    fh = logging.FileHandler(file_name)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.handlers = []
    # add handler to logger object
    logger.addHandler(fh)
    return logger
