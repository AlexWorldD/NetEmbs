# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
InventoriesAccount.py
Created by lex at 2019-03-26.
"""
from Simulation.Abstract.Account import *


class InventoriesAccount(Account):

    def __init__(self, env, name, initialStock=None):
        super().__init__(env, name, initialStock)

        if initialStock != None:
            self.container = simpy.Container(env, init=initialStock)

        self.purchaseObserver = InventoriesAccount.PurchaseObserver(self)
        self.salesObserver = InventoriesAccount.SalesObserver(self)

    def buyStock(self, numberOfStocks):

        c = numberOfStocks

        if c > 0:
            yield self.container.put(c)

        # DOESN'T WORK YET
        # if w > 0:
        # 	yield self.containerWrong.put(w)

        self.setChanged()
        for obs in self.notifyObservers():
            yield obs

    def sellStock(self, salesTransaction):
        c = salesTransaction

        if c > 0:
            yield self.container.get(c)

        # DOESN'T WORK YET
        # if w > 0 :
        # 	yield self.containerWrong.get(w)

        self.setChanged()
        for obs in self.notifyObservers():
            yield obs

    class PurchaseObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, buy):
            for obs in self.outer.buyStock(buy):
                yield obs

    class SalesObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, sell):
            for obs in self.outer.sellStock(sell):
                yield obs
