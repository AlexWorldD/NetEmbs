# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
FixedAssets.py
Created by lex at 2019-03-26.
"""

import random
import numpy as np
from Abstract.Transaction import Transaction
from Abstract.Process import Process
from Simulation.CONFIG import *


class AddFixedAssetsTransaction(Transaction):
    def __init__(self, name, env):
        super().__init__(name, env)
        self.fixed_assets = 0.0
        self.trade_pay = 0.0

    def newTransaction(self):
        unique_id = random.choice(VARIANTS)
        cur_transaction = self.new(postfix=unique_id)
        #         Generating amounts
        self.trade_pay = random.randint(TRANSACTIONS_LIMITS["FixedAssets"][0],
                                        TRANSACTIONS_LIMITS["FixedAssets"][1])

        # Add noise of type 2 when noisy financial accounts with very small amounts
        if NOISE["FixedAssets"]:
            noise = super().noise(self.trade_pay, unique_id, cur_transaction)
        else:
            noise = {"left": {"def": 0.0}, "right": {"def": 0.0}}
        self.fixed_assets = self.trade_pay + np.sum(list(noise["left"].values())) - np.sum(
            list(noise["right"].values()))
        self.addRecord("FixedAssets_" + str(unique_id), "FixedAssets", self.fixed_assets, cur_transaction)
        self.addRecord("TradePayables_" + str(unique_id), "TradePayables", -self.trade_pay,
                       cur_transaction)

        if PRINT:
            self.printTransaction()
        return self.trade_pay, self.fixed_assets, noise, str(unique_id)

    def printTransaction(self):
        print("AddFixedAssetsTransaction: TradePayables=", self.trade_pay, " -> FixedAssets=",
              self.fixed_assets)


class AddFixedAssetsProcess(Process):
    def __init__(self, name, env, transaction, term):
        """
        Initialize AddFixedAssetsProcess
        :param name: Process name
        :param env: SciPy environment
        :param transaction: SalesTransaction instance
        :param term: Terms of depreciation procedure
        """
        self.name = name
        self.Transaction = transaction
        self.env = env
        self.term = term
        if PRINT:
            print("Process ", self.name)
        #     Add Notifier for process and Observer
        self.transactionNotifier = super().TransactionNotifier(self)

    def start(self):
        while True:
            yield self.env.timeout(random.expovariate(1.0 / self.term))

            last_transaction = self.Transaction.newTransaction()

            self.transactionNotifier.setChanged()

            for obs in self.transactionNotifier.notifyObservers(last_transaction):
                yield obs
