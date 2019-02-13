from Process import *
from Transaction import *
import random

import numpy as np

class TaxDisbursementTransaction(Transaction):

	def __init__(self, name, env, taxpayablesAccount):
		Transaction.__init__(self, name, env)
		self.taxpayablesAccount = taxpayablesAccount
		self.debug = False
		self.name = name

	def newTransaction(self):
		
		#generate new transaction id
		transaction = self.new()

		totalTax = self.taxpayablesAccount.containerCorrect.level + self.taxpayablesAccount.containerWrong.level

		taxToPay = 0.8*totalTax

		c_f =  np.random.binomial(1,  1-max(min(1, 0.3 + random.gauss(0, 0.01)), 0), 1)[0]

		taxToPayCorrect = c_f*taxToPay
		taxToPayWrong 	= (1-c_f)*taxToPay


		self.addRecord("Tax", taxToPayCorrect, taxToPayWrong, transaction)
		self.addRecord("Cash", -taxToPayCorrect, taxToPayWrong, transaction)


		return [taxToPayCorrect, taxToPayWrong]


	def printTransaction(self, tax):
		if self.debug:
			print "Tax \t%d \n@cash \t\t %d" % (tax, tax)


class TaxDisbursementsProcess(Process):
	def __init__(self, env, name, transaction, periodicity):
		self.name = name
		self.env = env
		self.transaction = transaction

		self.periodicity = periodicity
		self.transactionNotifier = Process.TransactionNotifier(self)


	def start(self):

		while True:
			yield self.env.timeout(random.expovariate(1.0/self.periodicity))
			aTransaction = self.transaction.newTransaction()

			self.transactionNotifier.setChanged()

			for obs in self.transactionNotifier.notifyObservers(aTransaction):
				yield obs



class SalesTaxProcess(Process):

	def __init__(self, env, name):
		self.name 	= name
		self.env 	= env
		self.transactionNotifee = SalesTaxProcess.TransactionNotifeeTax(self)


	class TransactionNotifeeTax(Observer):
		def __init__(self, outer):
			self.outer = outer

		def update(self, observable, args):
			yield self.outer.env.timeout(0)

