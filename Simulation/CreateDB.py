# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
CreateDB.py
Created by lex at 2019-03-24.
"""

import sqlite3
from sqlite3 import Error
from Simulation.CONFIG import *


def cleanDB(db_file="FSN_Data.db"):
    try:
        if db_file is None:
            db_file = DB_PATH
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute('''DELETE FROM EntryRecords''')
        c.execute('''DELETE FROM JournalEntries''')
        c.execute('''DELETE FROM sqlite_sequence ''')
        conn.commit()
    except Error as e:
        print(e)
    finally:
        conn.close()


def connectDB(db_file=None):
    """
    Create a connection to Sqlite3 DB or create one if not exist
    :param db_file: path to DataBase with Financial data
    :return:
    """
    try:
        if db_file is None:
            db_file = DB_PATH
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS EntryRecords (
                         ID INTEGER PRIMARY KEY AUTOINCREMENT,
                         TID NUMERIC,
                         Name TEXT,
                         FA_Name TEXT,
                         Value NUMERIC
                        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS JournalEntries (
                         ID INTEGER PRIMARY KEY AUTOINCREMENT,
                         Time NUMERIC,
                         Name TEXT,
                         FA_Name TEXT
                        )''')
        conn.commit()
        print("DB successfully created!")
    except Error as e:
        print(e)
    finally:
        conn.close()
