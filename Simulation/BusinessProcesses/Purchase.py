# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
Purchase.py
Created by lex at 2019-03-26.
"""
import random
import numpy as np
from Abstract.Transaction import Transaction
from Abstract.Process import Process
from Abstract.Observer import Observer
from Simulation.CONFIG import *


class PurchaseTransaction(Transaction):
    def __init__(self, name, env):
        super().__init__(name, env)
        self.personal_expenses = 0.0
        self.other_expenses = 0.0
        self.prepaid_expenses = 0.0
        self.trade_pay = 0.0

    def newTransaction(self):
        unique_id = random.choice(VARIANTS)
        cur_transaction = self.new(postfix=unique_id)
        #         Generating amounts
        # TODO instead pure random values add selection from potential values, otherwise how can we define similar?
        self.personal_expenses = random.randint(TRANSACTIONS_LIMITS["Purchase"][0], TRANSACTIONS_LIMITS["Purchase"][1])
        self.other_expenses = random.randint(TRANSACTIONS_LIMITS["Purchase"][0], TRANSACTIONS_LIMITS["Purchase"][1])
        self.prepaid_expenses = random.randint(TRANSACTIONS_LIMITS["Purchase"][0], TRANSACTIONS_LIMITS["Purchase"][1])
        self.trade_pay = self.prepaid_expenses + self.personal_expenses + self.other_expenses

        # Add noise of type 2 when noisy financial accounts with very small amounts
        if NOISE["Purchase"]:
            noise = super().noise(self.trade_pay, unique_id, cur_transaction)
        else:
            noise = {"left": {"def": 0.0}, "right": {"def": 0.0}}
        self.trade_pay = self.trade_pay + np.sum(list(noise["left"].values())) - np.sum(list(noise["right"].values()))

        self.addRecord("PersonnelExpenses_" + str(unique_id), "PersonnelExpenses", -self.personal_expenses,
                       cur_transaction)
        self.addRecord("OtherExpenses_" + str(unique_id), "OtherExpenses", -self.other_expenses, cur_transaction)
        self.addRecord("PrepaidExpenses_" + str(unique_id), "PrepaidExpenses", -self.prepaid_expenses, cur_transaction)
        self.addRecord("TradePayables_" + str(unique_id), "TradePayables", self.trade_pay, cur_transaction)

        if PRINT:
            self.printTransaction()
        return self.trade_pay, self.personal_expenses, self.other_expenses, self.prepaid_expenses, noise, str(unique_id)

    def printTransaction(self):
        print("PurchaseTransaction: PersonalExpenses=", self.personal_expenses, ", OtherExpenses=", self.other_expenses,
              ", PrepaidExpenses=", self.prepaid_expenses, " -> TradePayables=", self.trade_pay)


class PurchaseProcess(Process):
    def __init__(self, name, env, transaction, term):
        """
        Initialize PayrollDisbursementProcess
        :param name: Process name
        :param env: SciPy environment
        :param transaction: SalesTransaction instance
        :param term: ?
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


class PurchaseInventoryProcess(Process):
    def __init__(self, name, env):
        """
        Initialize PayrollDisbursementProcess
        :param name: Process name
        :param env: SciPy environment
        """
        self.name = name
        self.env = env
        if PRINT:
            print("Process ", self.name)
        #     Add Notifier for process and Observer
        self.transactionNotifier = super().TransactionNotifier(self)
        self.lowStockTrigger = PurchaseInventoryProcess.Trigger(self)
        self.manualOrderTrigger = PurchaseInventoryProcess.ManualTrigger(self)

    def stockToLowProcess(self):
        numberOfStocksToBuy = 1000

        self.transactionNotifier.setChanged()
        for obs in self.transactionNotifier.notifyObservers(numberOfStocksToBuy):
            yield obs

    def manualOrderProcess(self):
        numberOfStocksToBuy = 1280

        self.transactionNotifier.setChanged()
        for obs in self.transactionNotifier.notifyObservers(numberOfStocksToBuy):
            yield obs

    class Trigger(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.stockToLowProcess():
                yield obs

    class ManualTrigger(Observer):
        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.manualOrderProcess():
                yield obs
