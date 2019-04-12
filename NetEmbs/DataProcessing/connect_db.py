# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
connect_db.py
Created by lex at 2019-03-24.
"""
import sqlite3
import pandas as pd
import logging


def upload_data_old(path_to_db='DataSimulation/Sample.db', limit=10):
    """
    Uploading data from EntryRecords database
    :param path_to_db: path to sqlite3 database
    :param limit: number of rows to be uploaded, None - upload all data
    :return: DataFrame with required structure
    """
    cnx = sqlite3.connect(path_to_db)
    # Loading data from db
    if isinstance(limit, int):
        db_data = pd.read_sql_query("SELECT * FROM EntryRecords LIMIT " + str(limit), cnx).drop(
            ["ID", "ValueIncorrect"], axis=1)
    else:
        db_data = pd.read_sql_query("SELECT * FROM EntryRecords", cnx).drop(["ID", "ValueIncorrect"], axis=1)
    # Split into two columns: Debit and Credit
    db_data["Debit"] = db_data["ValueCorrect"][db_data["ValueCorrect"] > 0.0]
    db_data["Credit"] = -db_data["ValueCorrect"][db_data["ValueCorrect"] < 0.0]
    db_data.fillna(0.0, inplace=True)
    db_data.rename(index=str, columns={"TID": "ID"}, inplace=True)
    return db_data


def upload_data(path_to_db='../Simulation/FSN_Data.db', limit=10):
    """
    Uploading data from EntryRecords database
    :param path_to_db: path to sqlite3 database
    :param limit: number of rows to be uploaded, None - upload all data
    :return: DataFrame with required structure
    """
    local_logger = logging.getLogger("NetEmbs.UploadData")
    local_logger.info("Connection to DataBase")
    cnx = sqlite3.connect(path_to_db)
    # Loading data from db
    if isinstance(limit, int):
        db_data = pd.read_sql_query("SELECT * FROM EntryRecords LIMIT " + str(limit), cnx).drop(
            ["ID"], axis=1)
    else:
        db_data = pd.read_sql_query("SELECT * FROM EntryRecords", cnx).drop(["ID"], axis=1)
        # Split into two columns: Debit and Credit
    local_logger.info("Data has been uploaded")
    db_data.rename(index=str, columns={"TID": "ID"}, inplace=True)
    return db_data[db_data["Value"] != 0.0]


def upload_JournalEntriesTruth(path_to_db='../Simulation/FSN_Data.db', limit=None):
    """
    Uploading data from EntryRecords database
    :param path_to_db: path to sqlite3 database
    :param limit: number of rows to be uploaded, None - upload all data
    :return: DataFrame with JournalEntries
    """
    local_logger = logging.getLogger("NetEmbs.UploadJournalEntries")
    local_logger.info("Connection to DataBase")
    cnx = sqlite3.connect(path_to_db)
    # Loading data from db
    if isinstance(limit, int):
        db_data = pd.read_sql_query("SELECT * FROM main.JournalEntries LIMIT " + str(limit), cnx)
    else:
        db_data = pd.read_sql_query("SELECT * FROM main.JournalEntries", cnx)
        # Split into two columns: Debit and Credit
    local_logger.info("Data has been uploaded")
    return db_data
