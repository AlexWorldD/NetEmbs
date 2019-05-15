# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
BadTransaction.py
Created by lex at 2019-05-15.
"""

import random
import numpy as np
from Abstract.Transaction import Transaction
from Abstract.Process import Process
from Simulation.CONFIG import *


class BadTransaction(Transaction):
    def __init__(self, tax_rate, env, name="both"):
        super().__init__(name, env)
        self.tax_rate = tax_rate
        self.trade_rec = 0.0
        self.revenue = 0.0
        self.tax = 0.0

    def newTransaction(self):
        unique_id = random.choice(VARIANTS)
        cur_transaction = self.new(postfix=unique_id)
        if self.name == "left":
            self.trade_rec = random.randint(1, 5)
        elif self.name == "right":
            self.revenue = random.randint(1, 5)
            self.tax = self.revenue * self.tax_rate
        self.addRecord("Revenue_" + str(unique_id), "Revenue", -self.revenue, cur_transaction)
        self.addRecord("Tax_" + str(unique_id), "Tax", -self.tax, cur_transaction)
        self.addRecord("TradeReceivables_" + str(unique_id), "TradeReceivables", self.trade_rec, cur_transaction)

        if PRINT:
            self.printTransaction()
        return self.trade_rec, self.revenue, self.tax, str(unique_id)

    def printTransaction(self):
        print("BadTransaction: Revenue=", self.revenue, ", Tax=", self.tax, " -> TR=", self.trade_rec)
