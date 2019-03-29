# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
CostOfSalesAccount.py
Created by lex at 2019-03-26.
"""

from Simulation.Abstract.Account import *


class CostOfSalesAccount(Account):
    def __init__(self, env, name, initialStock=None):
        super().__init__(env, name, initialStock)
        self.salesObserver = CostOfSalesAccount.SalesObserver(self)

    def processSales(self, transaction):
        c = transaction[0]

        if c > 0:
            yield self.container.put(c)
        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class SalesObserver(Observer):
        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, transaction):
            for obs in self.outer.processSales(transaction):
                yield obs
