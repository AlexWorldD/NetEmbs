from Process import *
from Transaction import *

import numpy as np

import random

class PuchaseProcess(Process):

	def __init__(self, env, name, term, transaction):
		self.env = env
		self.name = name
		self.term = term
		self.transaction = transaction
		self.transactionNotifier = Process.TransactionNotifier(self)
		

	def start(self):
		while True:
			yield self.env.timeout(random.expovariate(1.0/self.term))

			aTransaction = self.transaction.newTransaction()

			self.transactionNotifier.setChanged()

			for obs in self.transactionNotifier.notifyObservers(aTransaction):
				yield obs

class PurchaseTransaction(Transaction):

	def __init__(self, name, env):
		Transaction.__init__(self, name, env)

		self.name = name
		self.env = env
		self.debug = False


	def newTransaction(self):

		personnelExpenses = random.randint(1,100)
		otherExpenses = random.randint(1,100)
		prepaidExpense = random.randint(1,100)
		tradePayables = personnelExpenses + otherExpenses + prepaidExpense

		tid = self.new()

		correct_fraction = np.random.binomial(1,  1-max(min(1, 0.3 + random.gauss(0, 0.01)), 0), 1)[0]
		c2_f = np.random.binomial(1,  1-max(min(1, 0.3 + random.gauss(0, 0.01)), 0), 1)[0]
		c3_f = np.random.binomial(1,  1-max(min(1, 0.3 + random.gauss(0, 0.01)), 0), 1)[0]
		c4_f = np.random.binomial(1,  1-max(min(1, 0.3 + random.gauss(0, 0.01)), 0), 1)[0]

		self.addRecord("Trade Payables", tradePayables*(correct_fraction), (1-correct_fraction)*tradePayables, tid)
		self.addRecord("Other Expenses", -otherExpenses*c2_f, (1-c2_f)*otherExpenses, tid)
		self.addRecord("Prepaid Expenses", -c3_f*prepaidExpense, (1-c3_f)*prepaidExpense, tid)
		self.addRecord("Personnel Expenses", -personnelExpenses*c4_f, personnelExpenses*(1-c4_f), tid)

		self.printTransaction(tradePayables, otherExpenses, prepaidExpense, personnelExpenses)

		return [tradePayables*(correct_fraction), (1-correct_fraction)*tradePayables, otherExpenses*c2_f,  (1-c2_f)*otherExpenses, c3_f*prepaidExpense, (1-c3_f)*prepaidExpense, personnelExpenses*c4_f, personnelExpenses*(1-c4_f)]



	def printTransaction(self, tp, oe, pe, personnelExpenses):
		if self.debug:
			print "Trade Payables \t%s\n@Personnel Expenses \t\t %s\n@Other Expenses \t\t %s\n@Prepaid expenses \t\t %s" % (tp, personnelExpenses, oe, pe)


class PurchaseInventoryProcess(Process):

	def __init__(self, env, name):
		self.env 		= env
		self.name 		= name
		self.lowStockTrigger 	= PurchaseInventoryProcess.Trigger(self)
		self.manualOrderTrigger	= PurchaseInventoryProcess.ManualTrigger(self)


		self.transactionNotifier 	= Process.TransactionNotifier(self)
		

	def stockToLowProcess(self):
		numberOfStocksToBuy = 1000

		self.transactionNotifier.setChanged()
		for obs in self.transactionNotifier.notifyObservers([numberOfStocksToBuy,10]):
			yield obs


	def manualOrderProcess(self):
		numberOfStocksToBuy = 1234

		self.transactionNotifier.setChanged()
		for obs in self.transactionNotifier.notifyObservers([numberOfStocksToBuy,0]):
			yield obs


	class Trigger(Observer):

		def __init__(self,outer):
			self.outer = outer

		def update(self, observable, args):
			for obs in self.outer.stockToLowProcess():
				yield obs


	class ManualTrigger(Observer):
		def __init__(self, outer):
			self.outer = outer

		def update(self, observable, args):
			for obs in self.outer.manualOrderProcess():
				yield obs



