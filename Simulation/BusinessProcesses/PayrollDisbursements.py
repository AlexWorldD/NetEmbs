# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
PayrollDisbursements.py
Created by lex at 2019-03-26.
"""
import random
import numpy as np
from Abstract.Transaction import Transaction
from Abstract.Process import Process
from Abstract.Observer import Observer
from Simulation.CONFIG import *


class PayrollDisbursementsTransaction(Transaction):
    def __init__(self, name, env, eb_payables):
        super().__init__(name, env)
        self.cash = 0.0
        self.eb_pay = 0.0
        self.__eb_payables = eb_payables

    def newTransaction(self):
        self.eb_pay = self.__eb_payables.container.level
        unique_id = random.choice(VARIANTS)
        cur_transaction = self.new(postfix=unique_id)
        #         Generating amounts
        self.cash = self.eb_pay
        self.addRecord("EBPayables_" + str(unique_id), "EBPayables", self.eb_pay, cur_transaction)
        self.addRecord("Cash" + str(unique_id), "Cash", -self.cash, cur_transaction)

        if PRINT:
            self.printTransaction()
        return self.eb_pay, self.cash

    def printTransaction(self):
        print("PayrollDisbursementTransaction: EBPayables=", self.eb_pay, " -> Cash=", self.cash)


class PayrollDisbursementsProcess(Process):
    def __init__(self, name, env, transaction):
        """
        Initialize PayrollDisbursementProcess
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
        self.transactionNotifier = super().TransactionNotifier(self)
        self.Observer = PayrollDisbursementsProcess.PayrollObserver(self)

    def processPayrollDisbursement(self, transaction):
        last_transaction = self.Transaction.newTransaction()
        self.transactionNotifier.setChanged()

        for obs in self.transactionNotifier.notifyObservers(last_transaction):
            yield obs

    class PayrollObserver(Observer):
        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processPayrollDisbursement(args):
                yield obs
