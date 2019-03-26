# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
TradePayablesAccount.py
Created by lex at 2019-03-26.
"""
from Simulation.Abstract.Account import *


class TradePayablesAccount(Account):
    def __init__(self, env, name, initialStock=None):
        super().__init__(env, name, initialStock)
        self.purchaseObserver = TradePayablesAccount.PurchaseObserver(self)
        self.purchaseInventoryObserver = TradePayablesAccount.PurchaseInventoryObserver(self)
        self.fixedAssetsObserver = TradePayablesAccount.FixedAssetsObserver(self)

    def purchaseInventoryOrder(self, order):
        if order > 0:
            yield self.container.put(order)
        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    def processPurchase(self, purchase):
        trade_pay, personal_exp, other_exp, prepaid_exp, _, _ = purchase

        if trade_pay > 0:
            yield self.container.put(trade_pay)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    def processFixedAssets(self, assets):
        trade_pay, fixed_assets = assets

        if trade_pay > 0:
            yield self.container.put(trade_pay)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class FixedAssetsObserver(Observer):
        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processFixedAssets(args):
                yield obs

    class PurchaseObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processPurchase(args):
                yield obs

    class PurchaseInventoryObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, order):
            for obs in self.outer.purchaseInventoryOrder(order):
                yield obs
