# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
PrepaidExpensesAccount.py
Created by lex at 2019-03-26.
"""
from Simulation.Abstract.Account import *
from Simulation.FinancialAccounts.OtherExpensesAccount import *


class PrepaidExpensesAccount(Account):
    def __init__(self, env, name, initialStock=None):
        super().__init__(env, name, initialStock)

        self.purchaseObserver = OtherExpensesAccount.PurchaseObserver(self)

    def processPurchase(self, purchase):
        trade_pay, personal_exp, other_exp, prepaid_exp, _, _ = purchase

        if prepaid_exp > 0:
            self.container.put(prepaid_exp)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class PurchaseObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processPurchase(args):
                yield obs
