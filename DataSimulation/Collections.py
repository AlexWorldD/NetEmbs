from Transaction import *
from Process import *

import random 

import numpy as np

class CollectionsTransaction(Transaction):

	def __init__(self, name, env):
		Transaction.__init__(self, name, env)
		self.name = name
		self.env = env
		self.debug = False

	def newTransaction(self, salesTransaction):
		[tr, tr_w,  rev, rev_w,  tax, tax_w, label] = salesTransaction

		#generate new transaction ID
		transaction = self.new()

		self.addRecord("Trade Receivables", tr, tr_w, transaction)
		self.addRecord("Cash", -tr, -tr_w, transaction)

		self.printTransaction(tr)

		return [tr, tr_w]

	def printTransaction(self, tr):
		if self.debug:
			print "Cash \t %d \n @TradeRec \t\t\t %d" % (tr, tr)

class CollectionsProcess(Process):

	def __init__(self, env, name, transaction):
		self.name 						= name
		self.env 						= env
		self.transaction 				= transaction

		self.transactionNotifier		= Process.TransactionNotifier(self)
		self.collectionsProcessObserver = CollectionsProcess.CollectionsProcessObserver(self)


	def processSalesCollection(self, lastTransactionDetails):
		aTransacton = self.transaction.newTransaction(lastTransactionDetails)

		self.transactionNotifier.setChanged()

		for obs in self.transactionNotifier.notifyObservers(aTransacton):
			yield obs


	class CollectionsProcessObserver(Observer):
		def __init__(self, outer):
			self.outer = outer

		def update(self, observable, args):
			for obs in self.outer.processSalesCollection(args):
				yield obs