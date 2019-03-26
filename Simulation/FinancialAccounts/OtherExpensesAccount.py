# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
OtherExpensesAccount.py
Created by lex at 2019-03-26.
"""
from Simulation.Abstract.Account import *


class OtherExpensesAccount(Account):

    def __init__(self, env, name, initialStock=None):
        super().__init__(env, name, initialStock)

        self.purchaseObserver = OtherExpensesAccount.PurchaseObserver(self)

    def processPurchase(self, purchase):
        trade_pay, personal_exp, other_exp, prepaid_exp, _, _ = purchase

        if other_exp > 0:
            self.container.put(other_exp)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class PurchaseObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processPurchase(args):
                yield obs
