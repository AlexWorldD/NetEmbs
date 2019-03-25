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
            noise = super().noise(self.cost_of_sales, unique_id, cur_transaction)
        self.cost_of_sales = cost_of_sales - np.sum(list(noise["right"].values()))
        self.inventory = cost_of_sales - np.sum(list(noise["left"].values()))

        self.addRecord("CostOfSales_" + str(unique_id), "CostOfSales", self.cost_of_sales, cur_transaction)
        self.addRecord("inventory" + str(unique_id), "Inventory", -self.inventory, cur_transaction)

        if PRINT:
            self.printTransaction()

    def printTransaction(self):
        print("GoodsDeliveryTransaction: Inventory=", self.inventory, " -> CostOfSales=", self.cost_of_sales)
