# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
make_snapshot.py
Created by lex at 2019-07-29.
"""
from typing import Dict
from NetEmbs import CONFIG
import logging


def log_snapshot(cur_state: Dict, logger_name: str) -> None:
    """
    Helper function to log current state of FSN during walks
    Parameters
    ----------
    cur_state : Dict
            Snapshot of the state right before the exception
    logger_name : str
            Logger's name
    Returns
    -------
    None
    """
    local_logger = logging.getLogger(f"NetEmbs.{logger_name}")
    local_logger.error("Fatal ValueError during 1st sub-step", exc_info=True)
    local_logger.info("Snapshot" + str(cur_state))
    local_logger = logging.getLogger(f"{CONFIG.MAIN_LOGGER}.{logger_name}")
    local_logger.error("Fatal ValueError during 1st sub-step", exc_info=True)
    local_logger.info("Snapshot" + str(cur_state))
