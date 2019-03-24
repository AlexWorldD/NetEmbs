# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
Transaction.py
Created by lex at 2019-03-24.
"""

import sqlite3
from Simulation.CONFIG import *


class Transaction(object):

    def __init__(self, name, env):
        print("Transaction process ", name)
        self.conn = sqlite3.connect(DB_PATH)
        self.c = self.conn.cursor()
        self.env = env

    def addRecord(self, name, fa_name, value, tid):
        # Name - Unique name of Financial Account, eg. Product A and Product B;
        # FA_Name - Group names such as Revenue, Tax etc.
        self.c.execute("INSERT INTO EntryRecords (TID, Name, FA_Name, Value) VALUES (?, ?, ?, ?)",
                       (tid, name, fa_name, value))
        self.conn.commit()
        return self.c.lastrowid

        # return 1

    def new(self):
        self.c.execute("INSERT INTO JournalEntries (Time, Name, FA_Name) VALUES(?,?)",
                       (self.env.now, self.name, self.fa_name))
        self.conn.commit()
        return self.c.lastrowid
        # return 1
