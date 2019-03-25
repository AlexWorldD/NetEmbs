from Process import *
from Transaction import *

import random

import numpy as np

class DisbursementProcess(Process):

	def __init__(self, env, name, term, transaction):
		self.name = name
		self.env = env
		self.term = term
		self.transaction = transaction
		self.transactionNotifier = Process.TransactionNotifier(self)

	def start(self):

		while True:
			yield self.env.timeout(random.expovariate(1.0/self.term))

			aTransaction = self.transaction.newTransaction()

			self.transactionNotifier.setChanged()

			for obs in self.transactionNotifier.notifyObservers():
				yield obs


class DisbursementTrnsaction(Transaction):

	def __init__(self, name, env, tradepayables):
		Transaction.__init__(self, name, env)
		self.name = name
		self.env = env
		self.debug = False
		self._tradepayables = tradepayables

	def newTransaction(self):

		toPay = (self._tradepayables.containerCorrect.level +self._tradepayables.containerWrong.level)*0.75
		tid = self.new()

		c_f = np.random.binomial(1,  1-max(min(1, 0.3 + random.gauss(0, 0.01)), 0), 1)[0]


		self.addRecord("Trade Payables", toPay*c_f, toPay*(1-c_f), tid)
		self.addRecord("Cash", -toPay*c_f, -toPay*(1-c_f), tid)

		self.printTransaction(toPay, toPay)

		return [toPay*c_f, toPay*(1-c_f), toPay*c_f, toPay*(1-c_f)]



	def printTransaction(self, tp, cash):
		if self.debug:
			print "Trade Payables \t%s\n@Cash \t\t%s" % (tp, cash)
