from Process import *
from Transaction import *

import numpy as np
import random

class CosTransaction(Transaction):

	def __init__(self, name, env):
		Transaction.__init__(self, name, env)
		self.name = name
		self.debug = False


	def newTransaction(self, salesTransaction):
		[tr, tr_w,  rev, rev_w,  tax, tax_w, label] = salesTransaction

		#generate new transaction ID
		transaction = self.new()

		c_f = np.random.binomial(1,  1-max(min(1, 0.3 + random.gauss(0, 0.01)), 0), 1)[0]

		#generate values
		self.addRecord("Cost of Sales",c_f*rev, (1-c_f)*rev_w, transaction)
		self.addRecord("Inventory", -c_f*rev, -0.8*(1-c_f), transaction)

		self.printTransaction(0.8*rev,  0.8*rev)

		return [c_f*0.8*rev, (1-c_f)*0.8*rev_w]

	def printTransaction(self, cos, inv):
		if self.debug:
			print "Cost of Sales \t %d \n@Inventory \t\t\t %d \n" % (cos, inv)


class CosProcess(Process):
	def __init__(self, env, name, transaction):
		print "Process %s" % name

		self.name			= name
		self.env 			= env

		self.transactionNotifier		= Process.TransactionNotifier(self)
		self.transactionNotifee 		= CosProcess.TransactionNotifeeCOS(self)
		self.transaction 				= transaction

	class TransactionNotifeeCOS(Observer):
		def __init__(self, outer):
			self.outer = outer

		def update(self, observable, args):
			for obs in self.outer.processIncomingTransaction(args):
				yield obs


	def processIncomingTransaction(self, transaction):
		aTransaction = self.transaction.newTransaction(transaction)
		self.transactionNotifier.setChanged()

		for obs in self.transactionNotifier.notifyObservers(aTransaction):
			yield obs

