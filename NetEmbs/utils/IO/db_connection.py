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
    limit : int, default: None
            Rows to read from table

    Returns
    -------
        DataFrame
        DataFrame constructed from the passed in DataBase
    """
    cnx = sqlite3.connect(path)
    if limit is None:
        db_data = pd.read_sql_query(f"SELECT * FROM {table_name}", cnx)
    else:
        db_data = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {limit}", cnx)
    return db_data


def upload_data(path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Uploading data from EntryRecords table
    Parameters
    ----------
    path : str
            String path to the database
    limit : int, default: None
            Rows to read from table

    Returns
    -------
        DataFrame
        DataFrame constructed from the passed in DataBase
    """
    return upload_data_from(path, 'EntryRecords', limit).drop(["ID"], axis=1).rename(index=str, columns={"TID": "ID"})


def upload_journal_entries(path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Uploading data from JournalEntries table
    Parameters
    ----------
    path : str
            String path to the database
    limit : int, default: None
            Rows to read from table

    Returns
    -------
        DataFrame
        DataFrame constructed from the passed in DataBase
    """
    return upload_data_from(path, 'JournalEntries', limit).rename(index=str,
                                                                  columns={"FA_Name": "GroundTruth"})
