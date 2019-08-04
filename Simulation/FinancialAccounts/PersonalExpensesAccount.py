# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
PersonalExpensesAccount.py
Created by lex at 2019-03-26.
"""
from Simulation.Abstract.Account import *


class PersonnelExpensesAccount(Account):
    def __init__(self, env, name, initialStock=None):
        super().__init__(env, name, initialStock)
        self.payrollObserver = PersonnelExpensesAccount.PayrollObserver(self)
        self.purchaseObserver = PersonnelExpensesAccount.PurchaseObserver(self)

    def processPersonnelExpenses(self, payroll):
        salary, EB, tax = payroll

        if salary > 0:
            yield self.container.put(salary)
        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    def processPurchase(self, purchase):
        trade_pay, personal_exp, other_exp, prepaid_exp, _, _ = purchase

        if personal_exp > 0:
            yield self.container.put(personal_exp)
        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class PurchaseObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processPurchase(args):
                yield obs

    class PayrollObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processPersonnelExpenses(args):
                yield obs
