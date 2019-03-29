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
from Abstract.Observer import Observer
from Simulation.CONFIG import *


class CollectionsTransaction(Transaction):
    def __init__(self, name, env):
        super().__init__(name, env)
        self.trade_rec = 0.0
        self.cash = 0.0

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
        return self.trade_rec

    def printTransaction(self):
        print("CollectionsTransaction: TR=", self.trade_rec, " -> Cash=", self.cash)


class CollectionsProcess(Process):
    def __init__(self, name, env, transaction):
        self.name = name
        self.env = env
        self.Transaction = transaction
        if PRINT:
            print("Process ", self.name)
        #     Add Notifier for process and Observer
        self.transactionNotifier = super().TransactionNotifier(self)
        # TODO refactoring that part!!
        self.Observer = CollectionsProcess.CollectionsProcessObserver(self)

    def SalesCollection(self, last_transaction):
        cur_transaction = self.Transaction.newTransaction(last_transaction)
        self.transactionNotifier.setChanged()
        for obs in self.transactionNotifier.notifyObservers(cur_transaction):
            yield obs

    class CollectionsProcessObserver(Observer):
        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.SalesCollection(args):
                yield obs
