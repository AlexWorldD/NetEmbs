# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
TradeReceivablesAccount.py
Created by lex at 2019-03-26.
"""
from Simulation.Abstract.Account import *


class TradeReceivablesAccount(Account):
    def __init__(self, env, name, initialStock=None):
        super().__init__(env, name, initialStock)

        self.salesObserver = TradeReceivablesAccount.SalesObserver(self)
        self.collectionsObserver = TradeReceivablesAccount.CollectionsObserver(self)

    def salesOrder(self, lastTransactionDetails):
        tr, rev, tax, _, _ = lastTransactionDetails

        if tr > 0:
            yield self.container.put(tr)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    def collectionsOrder(self, collection):
        tr = collection
        if tr > 0:
            yield self.container.get(tr)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class SalesObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.salesOrder(args):
                yield obs

    class CollectionsObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.collectionsOrder(args):
                yield obs
