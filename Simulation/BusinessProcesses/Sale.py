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
from Simulation.CONFIG import *


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
        self.revenue = random.randint(TRANSACTIONS_LIMITS["Sales"][0], TRANSACTIONS_LIMITS["Sales"][1])
        self.tax = self.revenue * self.tax_rate
        # Add noise of type 1 when small diffs in amounts
        if NOISE["Sales"] and random.random() < NOISE_Type1["freq"]:
            self.tax += np.random.choice([-1.0, 1.0]) * random.uniform(0, NOISE_Type1["amplitude"]) * self.tax
        self.addRecord("Revenue_" + str(unique_id), "Revenue", -self.revenue, cur_transaction)
        self.addRecord("Tax_" + str(unique_id), "Tax", -self.tax, cur_transaction)
        # Add noise of type 2 when noisy financial accounts with very small amounts
        if NOISE["Sales"]:
            noise = super().noise(self.revenue, unique_id, cur_transaction)
        else:
            noise = {"left": {"def": 0.0}, "right": {"def": 0.0}}
        self.trade_rec = self.revenue + self.tax + np.sum(list(noise["left"].values())) - np.sum(
            list(noise["right"].values()))
        self.addRecord("TradeReceivables_" + str(unique_id), "TradeReceivables", self.trade_rec, cur_transaction)

        if PRINT:
            self.printTransaction()
        return self.trade_rec, self.revenue, self.tax, noise, str(unique_id)

    def printTransaction(self):
        print("SalesTransaction: Revenue=", self.revenue, ", Tax=", self.tax, " -> TR=", self.trade_rec)


class SalesProcess(Process):
    def __init__(self, name, env, transaction):
        """
        Initialize SalesProcess with name and SciPy environment
        :param name: Process name
        :param env: SciPy environment
        :param transaction: SalesTransaction instance
        """
        self.name = name
        self.Transaction = transaction
        self.env = env
        if PRINT:
            print("Process ", self.name)
        #     Add Notifier for process and Observer
        self.transactionNotifier = Process.TransactionNotifier(self)
        self.transactionObserver = Process.TransactionObserver(self)
        self.lastTransactionData = None

    def getTransactions(self, number):
        for _ in range(number):
            yield self.env.timeout(random.expovariate(1 / 4.0))
            self.lastTransactionData = self.Transaction.newTransaction()
            self.TransactionNotifier.setChanged(self)
            for obs in self.transactionNotifier.notifyObservers(self.lastTransactionData):
                yield obs
