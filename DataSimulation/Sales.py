from Transaction import *
from Process import *
from Event import *

import numpy as np

import random


class SalesTransaction(Transaction):

    def __init__(self, name, btw, env):
        Transaction.__init__(self, name, env)
        self.btw = btw
        self.name = name
        self.debug = False

    def newTransaction(self):
        # retrieve new transaction ID
        transaction = self.new()

        # generate amounts
        # TODO add type1 noise here
        rev = random.randint(10, 1000)
        tax = rev * self.btw
        tr = rev + tax

        # save records
        self.addRecord("Revenue", -rev, 0, transaction)
        self.addRecord("Tax", -tax, 0, transaction)
        self.addRecord("Trade receivables", tr, 0, transaction)

        c1_f = np.random.binomial(1, 1 - max(min(1, 0.3 + random.gauss(0, 0.01)), 0), 1)[0]
        c2_f = np.random.binomial(1, 1 - max(min(1, 0.3 + random.gauss(0, 0.01)), 0), 1)[0]
        c3_f = np.random.binomial(1, 1 - max(min(1, 0.3 + random.gauss(0, 0.01)), 0), 1)[0]

        self.printTransaction(tr, rev, tax)

        return [tr * c1_f, tr * (1 - c1_f), rev * c2_f, rev * (1 - c2_f), tax * c3_f, tax * (1 - c3_f), self.name]

    def printTransaction(self, tr, rev, tax):
        if self.debug:
            print "Trade receivables \t %d \n@Revenue \t\t\t %d \n@Tax \t\t\t\t %d \n" % (tr, rev, tax)


class SalesProcess(Process):

    def __init__(self, env, name, transaction):
        print "Process %s" % name

        self.name = name
        self.aTransaction = transaction
        self.env = env

        self.transactionNotifier = Process.TransactionNotifier(self)
        self.transactionNotifee = Process.TransactionNotifee(self)
        self.lastTransactionDetails = None

    def randomTransactions(self, runs, col):
        while True:
            yield self.env.timeout(random.expovariate(1 / 4.0))
            self.lastTransactionDetails = self.aTransaction.newTransaction()
            [tr, tr_w, rev, rev_w, tax, tax_w, label] = self.lastTransactionDetails

            self.transactionNotifier.setChanged()
            for obs in self.transactionNotifier.notifyObservers(self.lastTransactionDetails):
                yield obs
