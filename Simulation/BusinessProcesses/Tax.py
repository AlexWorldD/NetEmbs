# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
Tax.py
Created by lex at 2019-03-26.
"""
import random
import numpy as np
from Abstract.Transaction import Transaction
from Abstract.Process import Process
from Abstract.Observer import Observer
from Simulation.CONFIG import *


class TaxDisbursementsTransaction(Transaction):
    def __init__(self, name, env, tax_payables):
        super().__init__(name, env)
        self.cash = 0.0
        self.tax = 0.0
        self.__tax_payables = tax_payables

    def newTransaction(self):
        self.tax = self.__tax_payables.container.level * 0.8
        unique_id = random.choice(VARIANTS)
        cur_transaction = self.new(postfix=unique_id)
        #         Generating amounts
        self.cash = self.tax
        self.addRecord("Tax_" + str(unique_id), "Tax", self.tax, cur_transaction)
        self.addRecord("Cash_" + str(unique_id), "Cash", -self.cash, cur_transaction)

        if PRINT:
            self.printTransaction()
        return self.tax, self.cash

    def printTransaction(self):
        print("TaxDisbursementTransaction: Cash=", self.cash, " -> Tax=", self.tax)


class TaxDisbursementProcess(Process):
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


class SalesTaxProcess(Process):
    def __init__(self, name, env):
        self.name = name
        self.env = env
        if PRINT:
            print("Process ", self.name)
        #     Add Notifier for process and Observer
        self.Observer = SalesTaxProcess.TransactionNotifeeTax(self)

    class TransactionNotifeeTax(Observer):
        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            yield self.outer.env.timeout(0)
