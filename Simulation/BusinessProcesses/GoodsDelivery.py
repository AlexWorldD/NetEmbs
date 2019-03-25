# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
CostOfSales.py
Created by lex at 2019-03-25.
"""
import random
import numpy as np
from Abstract.Transaction import Transaction
from Abstract.Process import Process
from Abstract.Observer import Observer
from CONFIG import *


class GoodsDeliveryTransaction(Transaction):
    def __init__(self, name, env):
        super().__init__(name, env)
        self.cost_of_sales = 0.0
        self.inventory = 0.0

    def newTransaction(self, salesTransaction):
        _, cost_of_sales, _, noise, u_id = salesTransaction
        unique_id = random.choice(VARIANTS)
        cur_transaction = self.new(postfix=unique_id)
        #         Generating amounts
        if NOISE["GoodsDelivery"]:
            noise = super().noise(cost_of_sales, unique_id, cur_transaction)
        else:
            noise = {"left": {"def": 0.0}, "right": {"def": 0.0}}
        #     Update core processes regard the noisy FAs
        self.cost_of_sales = cost_of_sales - np.sum(list(noise["right"].values()))
        self.inventory = cost_of_sales - np.sum(list(noise["left"].values()))

        self.addRecord("CostOfSales_" + str(unique_id), "CostOfSales", self.cost_of_sales, cur_transaction)
        self.addRecord("inventory_" + str(unique_id), "Inventory", -self.inventory, cur_transaction)

        if PRINT:
            self.printTransaction()

        return self.inventory, self.cost_of_sales, noise, str(unique_id)

    def printTransaction(self):
        print("GoodsDeliveryTransaction: Inventory=", self.inventory, " -> CostOfSales=", self.cost_of_sales)


class GoodsDeliveryProcess(Process):
    def __init__(self, name, env, transaction):
        self.name = name
        self.env = env
        self.Transaction = transaction
        if PRINT:
            print("Process ", self.name)
        #     Add Notifier for process and Observer
        self.transactionNotifier = super().TransactionNotifier()
        self.Observer = GoodsDeliveryProcess.GoodsDeliveryProcessObserver(self)

    def IncomingTransaction(self, last_transaction):
        cur_transaction = self.Transaction.newTransaction(last_transaction)
        self.transactionNotifier.setChanged()
        for obs in self.transactionNotifier.notifyObservers(cur_transaction):
            yield obs

    class GoodsDeliveryProcessObserver(Observer):
        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processIncomingTransaction(args):
                yield obs
