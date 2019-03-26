# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
RevenueAccount.py
Created by lex at 2019-03-26.
"""
from Simulation.Abstract.Account import *


class RevenueAccount(Account):

    def __init__(self, env, name, initialStock=None):
        super().__init__(env, name, initialStock)

        self.salesObserver = RevenueAccount.SalesObserver(self)

    def processSales(self, lastTransactionDetails):
        trade_rec, revenue, tax, _, _ = lastTransactionDetails
        if revenue > 0:
            yield self.container.put(revenue)

    class SalesObserver(Observer):
        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processSales(args):
                yield obs
