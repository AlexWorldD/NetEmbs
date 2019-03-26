# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
Payroll.py
Created by lex at 2019-03-26.
"""

import random
import numpy as np
from Abstract.Transaction import Transaction
from Abstract.Process import Process
from Abstract.Observer import Observer
from CONFIG import *


class PayrollTransaction(Transaction):
    def __init__(self, name, env, monthly_salary):
        super().__init__(name, env)
        self.tax = 0.0
        self.EB = 0.0
        self.salary = monthly_salary

    def newTransaction(self):
        """
        Create Payroll transaction (add values to DB)
        :return: salary, EB, tax
        """
        unique_id = random.choice(VARIANTS)
        cur_transaction = self.new(postfix=unique_id)
        #         Generating amounts
        # TODO constant coefficients?
        self.tax = 0.21 * self.salary
        self.EB = 0.79 * self.salary
        self.addRecord("Tax_" + str(unique_id), "Tax", -self.tax, cur_transaction)
        self.addRecord("EBPayables_" + str(unique_id), "EBPayables", -self.EB, cur_transaction)
        self.addRecord("PersonnelExpenses_" + str(unique_id), "PersonnelExpenses", self.salary, cur_transaction)

        if PRINT:
            self.printTransaction()
        return self.salary, self.EB, self.tax

    def printTransaction(self):
        print("PersonalExpenses: EBPayables=", self.EB, ", Tax=", self.tax, " -> PersonalExpenses=", self.salary)


class PayrollProcess(Process):
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
        self.transactionNotifier = super().TransactionNotifier()

    def start(self):
        while True:
            yield self.env.timeout(4)

            last_transaction = self.Transaction.newTransaction()

            self.transactionNotifier.setChanged()
            for obs in self.transactionNotifier.notifyObservers(last_transaction):
                yield obs
