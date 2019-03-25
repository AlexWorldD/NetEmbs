# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
Sale.py
Created by lex at 2019-03-24.
"""
import random
import numpy as np
from Abstract.Transaction import Transaction
from Abstract.Process import Process
from CONFIG import *


class SalesTransaction(Transaction):
    def __init__(self, name, tax_rate, env):
        super().__init__(name, env)
        self.tax_rate = tax_rate
        self.trade_rec = 0.0
        self.revenue = 0.0
        self.tax = 0.0

    def newTransaction(self):
        unique_id = random.choice(VARIANTS)
        cur_transaction = self.new(postfix=unique_id)
        #         Generating amounts
        self.revenue = random.randint(TRANSACTIONS_LIMITS[0], TRANSACTIONS_LIMITS[1])
        self.tax = self.revenue * self.tax_rate
        # Add noise of type 1 when small diffs in amounts
        if NOISE["Sales"] and random.random() < NOISE_Type1["freq"]:
            self.tax += np.random.choice([-1.0, 1.0]) * random.uniform(0, NOISE_Type1["amplitude"]) * self.tax
        self.addRecord("Revenue_" + str(unique_id), "Revenue", -self.revenue, cur_transaction)
        self.addRecord("Tax_" + str(unique_id), "Tax", -self.tax, cur_transaction)
        # Add noise of type 2 when noisy financial accounts with very small amounts
        if NOISE["Sales"]:
            noise = super().noise(self.revenue, unique_id, cur_transaction)

        self.trade_rec = self.revenue + self.tax + np.sum(list(noise["left"].values())) - np.sum(
            list(noise["right"].values()))
        self.addRecord("TradeReceivables_" + str(unique_id), "TradeReceivables", self.trade_rec, cur_transaction)

        if PRINT:
            self.printTransaction()
        return self.trade_rec, self.revenue, self.tax, noise, str(unique_id)

    def printTransaction(self):
        print("SalesTransaction: Revenue=", self.revenue, ", Tax=", self.tax, " -> TR=", self.trade_rec)
