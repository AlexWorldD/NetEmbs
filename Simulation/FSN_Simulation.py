# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
FSN_Simulation.py
Created by lex at 2019-03-26.
"""
import simpy
from Simulation.CreateDB import *
from Simulation.Abstract.Transaction import *
from Simulation.BusinessProcesses.Sale import *
from Simulation.BusinessProcesses.Collections import *
from Simulation.BusinessProcesses.GoodsDelivery import *
from Simulation.BusinessProcesses.Purchase import *
from Abstract.Observer import Observer, Observable


class PayrollDisbursementEvent:

    def __init__(self, env, averageDisbursementTerm):
        self.env = env
        self.averageDisbursementTerm = averageDisbursementTerm
        self.payrolls = []

        self.payrollObserver = PayrollDisbursementEvent.PayrollDisbursementEventObserver(self)
        self.payrollObservable = PayrollDisbursementEvent.PayrollDisbursementEventObservable(self)

    def start(self):
        while True:
            # check if there are any disbursements ready?
            yield self.env.timeout(self.averageDisbursementTerm)
            if len(self.payrolls) > 0:
                eb = 0.0
                for e in self.payrolls:
                    eb += e

                if eb > 0:
                    del self.payrolls[:]  # empty list

                    self.payrollObservable.setChanged()

                    for obs in self.payrollObservable.notifyObservers(eb):
                        yield obs

    def processPayroll(self, transaction):

        salary, eb, tax = transaction
        self.payrolls.append(eb)
        yield self.env.timeout(0)

    class PayrollDisbursementEventObservable(Observable):

        def __init__(self, outer):
            super().__init__(self)
            self.outer = outer

    class PayrollDisbursementEventObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processPayroll(args):
                yield obs


class SalesCollectionEvent:
    def __init__(self, env, paymentTermInDays):
        self.env = env
        self.paymentTermInDays = paymentTermInDays
        self.collectionsObserver = SalesCollectionEvent.CollectionsObserver(self)
        self.collectionsObservable = SalesCollectionEvent.CollectionsObservable(self)
        self.transactionsQueue = []

    def start(self):
        yield self.env.timeout(14)
        while True:
            yield self.env.timeout(random.expovariate(1.0 / self.paymentTermInDays))

            if len(self.transactionsQueue) > 0:
                # get the first transaction in the queue
                last_transaction = self.transactionsQueue[0]

                self.transactionsQueue.remove(last_transaction)
                # Notif others that the transaction is ready to be processed
                yield self.env.process(self.notify(last_transaction))

    def notify(self, salesTransactionDetails):

        self.collectionsObservable.setChanged()

        for obs in self.collectionsObservable.notifyObservers(salesTransactionDetails):
            yield obs

    def processNewSalesDelay(self, salesTransactionDetails):
        # Add new transaction to the queue
        yield self.env.timeout(0)

        self.transactionsQueue.append(salesTransactionDetails)

    class CollectionsObservable(Observable):

        def __init__(self, outer):
            super().__init__(self)
            self.outer = outer

    class CollectionsObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processNewSalesDelay(observable.outer.lastTransactionDetails):
                yield obs


class ManualInventoryPurchaseEvent:

    def __init__(self, env):
        self.env = env
        self.observable = ManualInventoryPurchaseEvent.ManualInventoryPurchaseEventObservable(self)
        self.observer = ManualInventoryPurchaseEvent.ManualInventoryPurchaseEventObserver(self)

    class ManualInventoryPurchaseEventObservable(Observable):
        def __init__(self, outer):
            super().__init__(self)
            self.outer = outer

    def start(self):
        while True:
            # maybe this shouldn't be a time out but rather a probability of occurance.
            yield self.env.timeout(random.expovariate(1.0 / 2))
            self.observable.setChanged()

            for obs in self.observable.notifyObservers():
                yield obs

    class ManualInventoryPurchaseEventObserver(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            print("DummeManual")


class stockToLowEvent:

    def __init__(self, env, criticalLevel):
        self.criticalLevel = criticalLevel
        self.stockToLowObservable = stockToLowEvent.stockToLowObservable(self)
        self.stockToLowObservee = stockToLowEvent.stockToLowObservee(self)
        self.env = env

    def processLevel(self, level):
        if level < self.criticalLevel:
            self.stockToLowObservable.setChanged()
            for obs in self.stockToLowObservable.notifyObservers():
                yield obs

    class stockToLowObservable(Observable):
        def __init__(self, outer):
            super.__init__(self)
            self.outer = outer

    class stockToLowObservee(Observer):

        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processLevel(observable.containerCorrect.level):
                yield self.outer.env.timeout(0)

# //////////////// \\\\\\\\\\\\\\\