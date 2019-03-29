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
from Simulation.CONFIG import *


class DisbursementsTransaction(Transaction):
    def __init__(self, name, env, trade_payables):
        super().__init__(name, env)
        self.cash = 0.0
        self.trade_pay = 0.0
        self.__trade_payables = trade_payables

    def newTransaction(self):
        unique_id = random.choice(VARIANTS)
        cur_transaction = self.new(postfix=unique_id)
        #         Generating amounts
        self.trade_pay = self.__trade_payables.container.level * 0.75
        self.cash = -1.0 * self.trade_pay
        self.addRecord("TradePayables_" + str(unique_id), "TradePayables", -self.trade_pay, cur_transaction)
        self.addRecord("Cash" + str(unique_id), "Cash", self.cash, cur_transaction)

        if PRINT:
            self.printTransaction()
        return self.trade_pay, self.cash

    def printTransaction(self):
        print("DisbursementTransaction: Cash=", self.cash, " -> TradePayables=", self.trade_pay)


class DisbursementProcess(Process):
    def __init__(self, name, env, transaction, term):
        """
        Initialize DisbursementProcess
        :param name: Process name
        :param env: SciPy environment
        :param transaction: SalesTransaction instance
        :param term: Term of depreciation procedure
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
