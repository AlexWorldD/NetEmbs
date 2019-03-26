# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
EB_PayablesAccount.py
Created by lex at 2019-03-26.
"""
from Simulation.Abstract.Account import *


class EBPayablesAccount(Account):

    def __init__(self, env, name, initialStock=None):
        super().__init__(env, name, initialStock)
        self.payrollObserver = EBPayablesAccount.PayrollObserver(self)
        self.payrollDisbursementObserver = EBPayablesAccount.PayrollDisbursementObserver(self)

    def processPayroll(self, payroll):
        salary, EB, tax = payroll

        if EB > 0:
            yield self.container.put(EB)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    def processPayrollDisbursement(self, payroll):

        # small hack, cause of the time delay the amounts are two low if they come from the triggered event
        eb = self.container.level

        if eb > 0:
            yield self.container.get(eb)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class PayrollDisbursementObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processPayrollDisbursement(args):
                yield obs

    class PayrollObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processPayroll(args):
                yield obs
