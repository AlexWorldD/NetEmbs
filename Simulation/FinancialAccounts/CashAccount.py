# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
CashAccount.py
Created by lex at 2019-03-26.
"""
from Simulation.Abstract.Account import *


class CashAccount(Account):

    def __init__(self, env, name, initialStock=None):
        super().__init__(env, name, initialStock)
        self.collectionsObserver = CashAccount.CollectionsObserver(self)
        self.taxDisbursementObserver = CashAccount.TaxDisbursementObserver(self)
        self.payrollObserver = CashAccount.PayrollObserver(self)

    def processCollection(self, collection):

        tr = collection

        if tr > 0:
            yield self.container.put(tr)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    def processTaxDisbursement(self, tax):
        tax, cash = tax

        if tax > 0:
            yield self.container.get(tax)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    def processPayroll(self, payroll):
        salary, EB, tax = payroll

        if EB > 0:
            yield self.container.get(EB)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class PayrollObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processPayroll(args):
                yield obs

    class CollectionsObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processCollection(args):
                yield obs

    class TaxDisbursementObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processTaxDisbursement(args):
                yield obs
