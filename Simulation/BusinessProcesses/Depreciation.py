# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
Depreciation.py
Created by lex at 2019-03-26.
"""

import random
import numpy as np
from Abstract.Transaction import Transaction
from Abstract.Process import Process
from CONFIG import *


class DepreciationTransaction(Transaction):
    def __init__(self, name, env):
        super().__init__(name, env)
        self.fixed_assets = 0.0
        self.depreciation = 0.0

    def newTransaction(self):
        unique_id = random.choice(VARIANTS)
        cur_transaction = self.new(postfix=unique_id)
        #         Generating amounts
        self.depreciation = random.randint(TRANSACTIONS_LIMITS["Depreciation"][0],
                                           TRANSACTIONS_LIMITS["Depreciation"][1])

        # Add noise of type 2 when noisy financial accounts with very small amounts
        if NOISE["Depreciation"]:
            noise = super().noise(self.depreciation, unique_id, cur_transaction)
        else:
            noise = {"left": {"def": 0.0}, "right": {"def": 0.0}}
        self.fixed_assets = self.depreciation - np.sum(list(noise["left"].values())) + np.sum(
            list(noise["right"].values()))
        self.addRecord("FixedAssets_" + str(unique_id), "FixedAssets", -self.fixed_assets, cur_transaction)
        self.addRecord("DepreciationExpense_" + str(unique_id), "DepreciationExpense", self.depreciation,
                       cur_transaction)

        if PRINT:
            self.printTransaction()
        return self.depreciation, self.fixed_assets, noise, str(unique_id)

    def printTransaction(self):
        print("DepreciationTransaction: FixedAssets=", self.fixed_assets, " -> Depreciation Expense=",
              self.depreciation)


class DepreciationProcess(Process):
    def __init__(self, name, env, transaction, terms):
        """
        Initialize DepreciationProcess
        :param name: Process name
        :param env: SciPy environment
        :param transaction: SalesTransaction instance
        :param terms: Terms of depreciation procedure
        """
        self.name = name
        self.Transaction = transaction
        self.env = env
        self.terms = terms
        if PRINT:
            print("Process ", self.name)
        #     Add Notifier for process and Observer
        self.transactionNotifier = super().TransactionNotifier(self)

    def start(self):
        while True:
            yield self.env.timeout(self.terms)

            last_transaction = self.Transaction.newTransaction()

            self.transactionNotifier.setChanged()

            for obs in self.transactionNotifier.notifyObservers(last_transaction):
                yield obs
