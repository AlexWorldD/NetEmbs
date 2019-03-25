# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
Collections.py
Created by lex at 2019-03-25.
"""

import random
import numpy as np
from Abstract.Transaction import Transaction
from Abstract.Process import Process
from CONFIG import *


class CollectionsTransaction(Transaction):
    def __init__(self, name, env):
        super().__init__(name, env)
        self.trade_rec = 0.0
        self.cash = 0.0

    def newTransaction(self, salesTransaction):
        unique_id = random.choice(VARIANTS)
        cur_transaction = self.new(postfix=unique_id)
        #         Generating amounts
        self.revenue = random.randint(TRANSACTIONS_LIMITS[0], TRANSACTIONS_LIMITS[1])
        self.tax = self.revenue * self.tax_rate
        # Add noise of type 1 when small diffs in amounts
        if random.random() < NOISE_Type1["freq"]:
            self.tax *= np.random.choice([-1.0, 1.0]) * random.uniform(1, 1 + NOISE_Type1["amplitude"])
        self.addRecord("Revenue_" + str(unique_id), "Revenue", -self.revenue, cur_transaction)
        self.addRecord("Tax_" + str(unique_id), "Tax", -self.tax, cur_transaction)
        # Add noise of type 2 when noisy financial accounts with very small amounts
        noise = super().noise(self.revenue, unique_id, cur_transaction)

        self.trade_rec = self.revenue + self.tax + np.sum(noise["left"]) - np.sum(noise["right"])
        self.addRecord("TradeReceivables" + str(unique_id), "TradeReceivables", self.trade_rec, cur_transaction)

        if PRINT:
            self.printTransaction()

    def printTransaction(self):
        print("SalesTransaction: TR=", self.trade_rec, " -> Cash=", self.cash)
