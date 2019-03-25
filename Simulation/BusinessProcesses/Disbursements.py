# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
Disbursements.py
Created by lex at 2019-03-25.
"""

import random
import numpy as np
from Abstract.Transaction import Transaction
from Abstract.Process import Process
from CONFIG import *


class DisbursementsTransaction(Transaction):
    def __init__(self, name, env):
        super().__init__(name, env)
        self.cash = 0.0
        self.trade_pay = 0.0

    def newTransaction(self, salesTransaction):
        self.trade_rec, revenue, tax, noise, u_id = salesTransaction
        self.cash = self.trade_rec + np.sum(list(noise["right"].values()))
        noise["right"].pop("def")
        unique_id = random.choice(VARIANTS)
        cur_transaction = self.new(postfix=unique_id)
        #         Generating amounts

        self.addRecord("TradeReceivables_" + u_id, "TradeReceivables", -self.trade_rec, cur_transaction)
        for key, item in noise["right"].items():
            self.addRecord(key, key, -item, cur_transaction)

        self.addRecord("Cash" + str(unique_id), "Cash", self.cash, cur_transaction)

        if PRINT:
            self.printTransaction()

    def printTransaction(self):
        print("CollectionTransaction: TR=", self.trade_rec, " -> Cash=", self.cash)