# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
db_connection.py
Created by lex at 2019-03-24.
"""
import sqlite3
import pandas as pd
from typing import Optional


def upload_data_from(path: str, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Uploading data from the given SQlite Database
    Parameters
    ----------
    path : str
            String path to the database
    table_name : str
            String name of the table to upload data
    limit : int: None
            Rows to read from table

    Returns
    -------
        DataFrame
        DataFrame constructed from the passed in DataBase
    """
    cnx = sqlite3.connect(path)
    if limit is None:
        db_data = pd.read_sql_query(f"SELECT * FROM {table_name}", cnx).drop(["ID"], axis=1)
    else:
        db_data = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", cnx).drop(["ID"], axis=1)
    db_data.rename(index=str, columns={"TID": "ID"}, inplace=True)
    return db_data


def upload_data(path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Uploading data from EntryRecords table
    Parameters
    ----------
    path : str
            String path to the database
    limit : int: None
            Rows to read from table

    Returns
    -------
        DataFrame
        DataFrame constructed from the passed in DataBase
    """
    return upload_data_from(path, 'EntryRecords', limit).rename(index=str, columns={"TID": "ID"})


def upload_journal_entries(path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Uploading data from JournalEntries table
    Parameters
    ----------
    path : str
            String path to the database
    limit : int: None
            Rows to read from table

    Returns
    -------
        DataFrame
        DataFrame constructed from the passed in DataBase
    """
    return upload_data_from(path, 'JournalEntries', limit).rename(index=str,
                                                                  columns={"FA_Name": "GroundTruth"})

# def upload_data2(path_to_db='../Simulation/FSN_Data.db', limit=997, logger_name="NetEmbs"):
#     """
#     Uploading data from EntryRecords database
#     :param path_to_db: path to sqlite3 database
#     :param limit: number of rows to be uploaded, None - upload all data
#     :param logger_name: Name of logger to be used
#     :return: DataFrame with required structure
#     """
#     local_logger = logging.getLogger(logger_name + ".UploadData")
#     local_logger.info("Connection to DataBase")
#     cnx = sqlite3.connect(path_to_db)
#     # Loading data from db
#     if isinstance(limit, int):
#         db_data = pd.read_sql_query("SELECT * FROM EntryRecords LIMIT " + str(limit), cnx).drop(
#             ["ID"], axis=1)
#     else:
#         db_data = pd.read_sql_query("SELECT * FROM EntryRecords", cnx).drop(["ID"], axis=1)
#         # Split into two columns: Debit and Credit
#     local_logger.info("Data has been uploaded")
#     db_data.rename(index=str, columns={"TID": "ID"}, inplace=True)
#     return db_data
#     # return db_data[db_data["Value"] != 0.0]
#
#
# def upload_JournalEntriesTruth(path_to_db='../Simulation/FSN_Data.db', limit=None):
#     """
#     Uploading data from EntryRecords database
#     :param path_to_db: path to sqlite3 database
#     :param limit: number of rows to be uploaded, None - upload all data
#     :return: DataFrame with JournalEntries
#     """
#     local_logger = logging.getLogger("NetEmbs.UploadJournalEntries")
#     local_logger.info("Connection to DataBase")
#     cnx = sqlite3.connect(path_to_db)
#     # Loading data from db
#     if isinstance(limit, int):
#         db_data = pd.read_sql_query("SELECT * FROM main.JournalEntries LIMIT " + str(limit), cnx)
#     else:
#         db_data = pd.read_sql_query("SELECT * FROM main.JournalEntries", cnx)
#         # Split into two columns: Debit and Credit
#     # TODO add renaming
#     local_logger.info("Data has been uploaded")
#     return db_data.rename(index=str, columns={"FA_Name": "GroundTruth"})
