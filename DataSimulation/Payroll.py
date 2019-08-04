from Process import *
from Transaction import *

import random

import numpy as np

class PayrollDisbursementsProcess(Process):

	def __init__(self, env, name, transaction):
		self.name = name
		self.env = env
		self.atransaction = transaction
		self.payrollObserver = PayrollDisbursementsProcess.PayrollObserver(self)
		self.transactionNotifier = Process.TransactionNotifier(self)


	def processPayrollDisbursement(self, transaction):
		aTransaction = self.atransaction.newTransaction(transaction)
		self.transactionNotifier.setChanged()

		for obs in self.transactionNotifier.notifyObservers(aTransaction):
			yield obs

	class PayrollObserver(Observer):
		def __init__(self, outer):
			self.outer = outer


		def update(self, observable, args):
			for obs in self.outer.processPayrollDisbursement(args):
				yield obs


class PayrollDisbursementTransaction(Transaction):
	def __init__(self, name, env, aEBPayablesAccount):
		Transaction.__init__(self, name, env)
		self.name = name
		self.env = env
		self.debug = False
		self.ebpayables = aEBPayablesAccount


	def newTransaction(self, transaction):
		
		
		eb = self.ebpayables.containerCorrect.level
		eb_w = self.ebpayables.containerWrong.level

		#Generate transaction ID
		tid = self.new()

		self.addRecord("EB Payable", eb, eb_w, tid)
		self.addRecord("Cash", -eb, -eb_w, tid)

		self.printTransaction(eb, eb)

		return [eb, eb_w]

	def printTransaction(self, eb, cash):
		if self.debug:
			print "EB Payable \t %s\n@Cash \t\t\t%s" % (eb, cash)

class PayrollTransaction(Transaction):
	def __init__(self, name, env, monthlySalary):
		Transaction.__init__(self, name, env)
		self.name = name
		self.env = env
		self.monthlySalary = monthlySalary
		self.debug = False

	def newTransaction(self):

		#generate transaction id
		transaction = self.new()

		#salary is set to 15000
		salary 	= self.monthlySalary
		EB 		= 0.79*salary
		tax 	= 0.21*salary

		self.addRecord("Tax", -tax, -0, transaction)
		self.addRecord("EB payable", -EB, -0, transaction)
		self.addRecord("Personnel Expenses", salary, 0, transaction)

		self.printTransaction(salary)

		cf_1 = np.random.binomial(1,  1-max(min(1, 0.3 + random.gauss(0, 0.01)), 0), 1)[0]
		cf_2 = np.random.binomial(1,  1-max(min(1, 0.3 + random.gauss(0, 0.01)), 0), 1)[0]
		cf_3 = np.random.binomial(1,  1-max(min(1, 0.3 + random.gauss(0, 0.01)), 0), 1)[0]


		return [salary*cf_1, salary*(1-cf_1), EB*cf_2, EB*(1-cf_2), tax*cf_3, tax*(1-cf_3)]

	def printTransaction(self, salary):
		if self.debug:
			print "Personnel Expenses \t%d \n@EB Payable \t\t:%d\n@Tax Payables: \t\t %d" % (salary, 0.79*salary, 0.21*salary)


class PayrollProcess(Process):

	def __init__(self, env, name, transaction):
		self.env 					= env
		self.name 					= name
		self.transaction 			= transaction
		self.transactionNotifier 	= Process.TransactionNotifier(self)

	def start(self):
		while True:
			yield self.env.timeout(4)

			aTransaction = self.transaction.newTransaction()

			self.transactionNotifier.setChanged()
			for obs in self.transactionNotifier.notifyObservers(aTransaction):
				yield obs
