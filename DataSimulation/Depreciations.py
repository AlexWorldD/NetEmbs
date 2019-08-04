from Process import *
from Transaction import *

import random
import numpy as np

class DepreciationProcess(Process):

	def __init__(self, env, name, terms, transaction):
		self.name = name
		self.env = env
		self.terms = terms
		self.transaction = transaction

		self.transactionNotifier = DepreciationProcess.TransactionNotifier(self)


	def start(self):
		while True:
			yield self.env.timeout(self.terms)

			aTransaction = self.transaction.newTransaction()

			self.transactionNotifier.setChanged()


			for obs in self.transactionNotifier.notifyObservers(aTransaction):
				yield obs


class DepreciationTransaction(Transaction):
	def __init__(self, name, env):
		Transaction.__init__(self, name, env)

		self.name = name
		self.env  = env
		self.debug = False

	def newTransaction(self):

		depr = random.randint(1000,10000)

		tid = self.new()

		c_f = np.random.binomial(1,  1-max(min(1, 0.3 + random.gauss(0, 0.01)), 0), 1)[0]

		self.addRecord("Depreciation Expense", depr*c_f, depr*(1-c_f), tid)
		self.addRecord("Fixed Assets", -depr*c_f, -depr*(1-c_f), tid)


		self.printTransaction(depr)

		# depr, depr_w, fix, fix_w
		return [depr*c_f, depr*(1-c_f), depr*c_f, depr*(1-c_f)]

	def printTransaction(self, depr):
		if self.debug:
			print "Depreciation Expense \t%s\n@Fixed Assets \t\t%s\n" % (depr, depr)

