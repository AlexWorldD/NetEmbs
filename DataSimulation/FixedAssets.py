from Process import *
from Transaction import *

import random

import numpy as np

class AddToFixedAssetsProcess(Process):

	def __init__(self, env, name, term, transaction):
		self.name = name
		self.env = env
		self.transaction = transaction
		self.term = term
		self.transactionNotifier = Process.TransactionNotifier(self)


	def start(self):
		while True:
			yield self.env.timeout(random.expovariate(1.0/self.term))
			aTransaction = self.transaction.newTransaction()
			self.transactionNotifier.setChanged()

			for obs in self.transactionNotifier.notifyObservers(aTransaction):
				yield obs

class FixedAssetsTransaction(Transaction):

	def __init__(self, name, env):
		Transaction.__init__(self, name, env)

		self.name = name
		self.env  = env
		self.debug = False

	def newTransaction(self):

		additions = random.randint(1000,10000)

		tid = self.new()

		cf_1 = np.random.binomial(1,  1-max(min(1, 0.3 + random.gauss(0, 0.01)), 0), 1)[0]

		self.addRecord("Fixed Assets", additions*cf_1, additions*(1-cf_1), tid)
		self.addRecord("Trade Payables", -additions*cf_1, -additions*(1-cf_1), tid)

		self.printTransaction(additions)

		# fix, fix_w, tp, tp_w
		return [additions*cf_1, additions*(1-cf_1), additions*cf_1, additions*(1-cf_1)]

	def printTransaction(self, additions):
		if self.debug:
			print "Fixed assets \t%s\n@Trade Payables \t\t%s\n" % (additions, additions)