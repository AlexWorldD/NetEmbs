# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
TaxPayablesAccount.py
Created by lex at 2019-03-26.
"""
from Simulation.Abstract.Account import *


class TaxPayablesAccount(Account):
    def __init__(self, env, name, initialStock=None):
        super().__init__(env, name, initialStock)
        self.salesObserver = TaxPayablesAccount.SalesObserver(self)
        self.taxDisbursementObserver = TaxPayablesAccount.TaxDisbursementObserver(self)
        self.payrollObserver = TaxPayablesAccount.PayrollObserver(self)

    def processTaxExpenses(self, payroll):
        salary, EB, tax = payroll

        if tax > 0:
            yield self.container.put(tax)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class PayrollObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processTaxExpenses(args):
                yield obs

    def salesOrder(self, lastTransactionDetails):
        tr, rev, tax, _, _ = lastTransactionDetails

        if tax > 0:
            yield self.container.put(tax)
        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    def processTaxDisbursement(self, tax):
        # [tax_c, tax_w] = tax

        tax = self.container.level * 0.8

        if tax > 0:
            yield self.container.get(tax)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class TaxDisbursementObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processTaxDisbursement(args):
                yield obs

    class SalesObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.salesOrder(args):
                yield obs
