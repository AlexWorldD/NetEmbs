import simpy
import random

from FinancialStatement import *

from Observer import *
from Process import *
from Account import *
from Event import *

from Sales import *
from CostOfSales import * 
from Collections import *
from Depreciations import *
from Disbursements import *
from FixedAssets import *
from Purchases import *
from Periodics import *
from Tax import *	
from Payroll import *


## NOTICE: Yield from is a python 3.0 feature therefore use
# for a in yieldfromfunction():
# 	yield a
# which is similar as yield from yielfromfunction()


	
#event is an object that triggers a process. 	
#stock to low triggers purchase event
#order triggers purchase event

class PayrollDisbursementEvent(Event):

	def __init__(self, env, averageDisbursementTerm):
		self.env = env
		self.averageDisbursementTerm = averageDisbursementTerm
		self.payrolls = []


		self.payrollObserver 	= PayrollDisbursementEvent.PayrollDisbursementEventObserver(self)
		self.payrollObservable 	= PayrollDisbursementEvent.PayrollDisbursementEventObservable(self)


	def start(self):
		while True:
			#check if there are any disbursements ready?
			yield self.env.timeout(self.averageDisbursementTerm)
			if len(self.payrolls) > 0 :
				eb_c = 0
				eb_w = 0
				for c,w in self.payrolls:
					eb_c += c
					eb_w += w

				if eb_c > 0:
					del self.payrolls[:] #empty list

					self.payrollObservable.setChanged()

					for obs in self.payrollObservable.notifyObservers([eb_c, eb_w]):
						yield obs


	def processPayroll(self, transaction):

		[sal, sal_w, eb, eb_w, tax, tax_w] = transaction
		tot = [eb, eb_w]
		self.payrolls.append(tot)
		yield self.env.timeout(0)


	class PayrollDisbursementEventObservable(Observable):

		def __init__(self, outer):
			Observable.__init__(self)
			self.outer = outer

	class PayrollDisbursementEventObserver(Observer):

		def __init__(self, outer):
			self.outer = outer

		def update(self, observable, args):
			for obs in  self.outer.processPayroll(args):
				yield obs


class SalesCollectionEvent(Event):
	def __init__(self, env, paymentTermInDays):
		self.env = env
		self.paymentTermInDays 		= paymentTermInDays
		self.collectionsObserver 	= SalesCollectionEvent.CollectionsObserver(self)
		self.collectionsObservable 	= SalesCollectionEvent.CollectionsObservable(self) 
		self.transactionsQueue 		= []

	def start(self):
		yield self.env.timeout(14)
		while True:
			yield self.env.timeout(random.expovariate(1.0/self.paymentTermInDays))

			if len(self.transactionsQueue) > 0:
				#get the first transaction in the queue
				lastTransactionDetails = self.transactionsQueue[0]

				self.transactionsQueue.remove(lastTransactionDetails)
				#Notif others that the transaction is ready to be processed
				yield self.env.process(self.notify(lastTransactionDetails))

	def notify(self, salesTransactionDetails):

		self.collectionsObservable.setChanged()

		for obs in self.collectionsObservable.notifyObservers(salesTransactionDetails):
			yield obs

	def processNewSalesDelay(self, salesTransactionDetails):
		#Add new transaction to the queue
		yield self.env.timeout(0)

		self.transactionsQueue.append(salesTransactionDetails)


	class CollectionsObservable(Observable):

		def __init__(self, outer):
			Observable.__init__(self)
			self.outer = outer

	class CollectionsObserver(Observer):

		def __init__(self, outer):
			self.outer = outer

		def update(self, observable, args):
			for obs in self.outer.processNewSalesDelay(observable.outer.lastTransactionDetails):
				yield obs


class ManualInventoryPurchaseEvent(Event):

	def __init__(self, env):
		self.env = env
		self.observable = ManualInventoryPurchaseEvent.ManualInventoryPurchaseEventObservable(self)
		self.observer 	= ManualInventoryPurchaseEvent.ManualInventoryPurchaseEventObserver(self)

	class ManualInventoryPurchaseEventObservable(Observable):
		def __init__(self, outer):
			Observable.__init__(self)
			self.outer = outer


	def start(self):
		while True:
			#maybe this shouldn't be a time out but rather a probability of occurance. 
			yield self.env.timeout(random.expovariate(1.0/2))
			self.observable.setChanged()

			for obs in self.observable.notifyObservers():
				yield obs

	class ManualInventoryPurchaseEventObserver(Observer):

		def __init__(self, outer):
			self.outer = outer

		def update(self, observable, args):
			print "dummy"


		


class stockToLowEvent(Event):

	def __init__(self, env, criticalLevel):
		self.criticalLevel 			= criticalLevel
		self.stockToLowObservable 	= stockToLowEvent.stockToLowObservable(self)
		self.stockToLowObservee   	= stockToLowEvent.stockToLowObservee(self)
		self.env 					= env


	def processLevel(self, level):
		if level < self.criticalLevel:
			self.stockToLowObservable.setChanged()
			for obs in self.stockToLowObservable.notifyObservers():
				yield obs

	class stockToLowObservable(Observable):
		def __init__(self, outer):
			Observable.__init__(self)
			self.outer = outer


	class stockToLowObservee(Observer):

		def __init__(self, outer):
			self.outer = outer

		def update(self, observable, args):
			for obs in self.outer.processLevel(observable.containerCorrect.level):
				yield self.outer.env.timeout(0)

class BussinessSimulation(object):

	def simulate(self):

		env 					= simpy.Environment()

		revenueAccount 			= RevenueAccount(env, simpy, "Revenue")
		cosAccount				= CostOfSalesAccount(env, simpy, "Cost of Sales")
		taxPayablesAccount 		= TaxPayablesAccount(env, simpy, "Tax Payables")
		tradeReceivablesAccount = TradeReceivablesAccount(env, simpy, "Trade receivables")
		inventoriesAccount		= InventoriesAccount(env, simpy, "Inventories", 1200)
		EBPayableAccount 		= EBPayablesAccount(env, simpy, "EB Payables")
		personnelAccount		= PersonnelExpensesAccount(env, simpy, "Personnel Expenses")
		cashAccount 			= CashAccount(env, simpy, "Cash")
		deprExpenseAccount 		= DepreciationAccount(env, simpy, "Depreciation Expense")
		fixedAssetsAccount 		= FixedAssetsAccount(env, simpy, "Fixed Assets")
		tradePayablesAccount 	= TradePayablesAccount(env, simpy, "Trade Payables")
		otherExpensesAccount 	= OtherExpensesAccount(env, simpy, "Other Expenses")
		prepaidExpensesAccount 	= PrepaidExpensesAccount(env, simpy, "Prepaid Expenses")
		accrualsAccount 		= Account(env, simpy, "Accruals")
		LTDebtAccount			= Account(env, simpy, "LT Debt")
		interestExpense 		= Account(env, simpy, "Interest expense")


		salesTransactionHigh	= SalesTransaction("Sales 21 btw", 0.21, env) #Sales with high tax percentage
		salesTransactionLow		= SalesTransaction("Sales 6 btw", 0.06, env) #sales with low tax percentage

		disbursementTransactionTax			= TaxDisbursementTransaction("Tax disbursement", env, taxPayablesAccount)

		cosTransaction 			= CosTransaction("Cost of Sales", env)
		collectionTransaction	= CollectionsTransaction("Collections", env)
		payrollTransaction		= PayrollTransaction("Payroll", env, 1500)
		payrollDisbursementTransaction = PayrollDisbursementTransaction("Payroll Disbursement", env, EBPayableAccount)
		purchaseTransaction = PurchaseTransaction("Purchase", env)
		disbursementTransaction = DisbursementTransaction("Disbursement", env, tradePayablesAccount)
		fixesAssetsTransaction = FixedAssetsTransaction("Fixed Assets", env)
		depreciationTransaction = DepreciationTransaction("Depreciation", env)


		salesHigh 		= SalesProcess(env, "Sales high", salesTransactionHigh)
		salesLow 		= SalesProcess(env, "Sales low", salesTransactionLow)

		cos 			= CosProcess(env, "Cost of Sales", cosTransaction)

		salesTax 		= SalesTaxProcess(env, "Sales tax")

		salesTaxDisbur	= TaxDisbursementsProcess(env, "Tax Disbursement", disbursementTransactionTax, 2)


		purchaseInv 	= PurchaseInventoryProcess(env, "Purchase inventory Process")

		collections 	= CollectionsProcess(env, "Collections Process", collectionTransaction)

		payroll 		= PayrollProcess(env, "Payroll Process", payrollTransaction)

		payrollDisbursement = PayrollDisbursementsProcess(env, "Payroll Disbursement Process", payrollDisbursementTransaction)

		purchaseProcess = PuchaseProcess(env, "Purchanse Process", 20, purchaseTransaction)

		disbursementProcess = DisbursementProcess(env, "Disbursement Process", 10, disbursementTransaction)

		fixedAssetsProcess = AddToFixedAssetsProcess(env, "Fixed assets", 10, fixesAssetsTransaction)

		depreciationProcess = DepreciationProcess(env, "Depreciation Process", 10, depreciationTransaction)


		#events stock to low triggers purchase process
		stockToLow 				= stockToLowEvent(env, 1000)
		#observe the current stock
		inventoriesAccount.addObserver(stockToLow.stockToLowObservee) 


		#send to low signal to purchase inventory
		stockToLow.stockToLowObservable.addObserver(purchaseInv.lowStockTrigger) 



		#manual inventory purchase random irregular
		# manualOrderEvent = ManualInventoryPurchaseEvent(env)
		# manualOrderEvent.observable.addObserver(purchaseInv.manualOrderTrigger)



		#the revenue accounts wants to be observer of all processes that changes the revenue, i.e., sales
		salesLow.transactionNotifier.addObserver(revenueAccount.salesObserver)
		salesHigh.transactionNotifier.addObserver(revenueAccount.salesObserver)

		salesLow.transactionNotifier.addObserver(taxPayablesAccount.salesObserver)
		salesHigh.transactionNotifier.addObserver(taxPayablesAccount.salesObserver)

		salesLow.transactionNotifier.addObserver(tradeReceivablesAccount.salesObserver)
		salesHigh.transactionNotifier.addObserver(tradeReceivablesAccount.salesObserver)




		# Sales is done, book the Cost of Sales to Inventory
		cos.transactionNotifier.addObserver(cosAccount.salesObserver)
		cos.transactionNotifier.addObserver(inventoriesAccount.salesObserver)

		#the inventories account is an observer of any new purchase transaction
		purchaseInv.transactionNotifier.addObserver(inventoriesAccount.purchaseObserver)

		#observe for new inventory purchase orders, if so then update the tradePayables account
		purchaseInv.transactionNotifier.addObserver(tradePayablesAccount.purchaseInventoryObserver)

		#sale with payment term triggers collections (14 days payment term)
		collectionsEvent 		= SalesCollectionEvent(env, 2)
		#observe the sales process for new sales
		salesLow.transactionNotifier.addObserver(collectionsEvent.collectionsObserver)
		salesHigh.transactionNotifier.addObserver(collectionsEvent.collectionsObserver)

		#add the collections process as an observer of the collectionsEvent (time-delayed sales)
		collectionsEvent.collectionsObservable.addObserver(collections.collectionsProcessObserver)

		collections.transactionNotifier.addObserver(cashAccount.collectionsObserver)
		collections.transactionNotifier.addObserver(tradeReceivablesAccount.collectionsObserver)


		# add observer for the sales transactions
		salesHigh.transactionNotifier.addObserver(cos.transactionNotifee)
		salesLow.transactionNotifier.addObserver(cos.transactionNotifee)

		salesHigh.transactionNotifier.addObserver(salesTax.transactionNotifee)
		salesLow.transactionNotifier.addObserver(salesTax.transactionNotifee)


		#Tax disbursement process
		salesTaxDisbur.transactionNotifier.addObserver(cashAccount.taxDisbursementObserver)
		salesTaxDisbur.transactionNotifier.addObserver(taxPayablesAccount.taxDisbursementObserver)


		#Payroll observers
		payroll.transactionNotifier.addObserver(EBPayableAccount.payrollObserver)
		payroll.transactionNotifier.addObserver(personnelAccount.payrollObserver)
		payroll.transactionNotifier.addObserver(taxPayablesAccount.payrollObserver)

		payrollDisbursementEvent = PayrollDisbursementEvent(env, 2)
		payroll.transactionNotifier.addObserver(payrollDisbursementEvent.payrollObserver) #is a payroll transaction takes place the delayed event wants to be notified
		payrollDisbursementEvent.payrollObservable.addObserver(payrollDisbursement.payrollObserver) #notify the payroll disbursement process if the event happened

		payrollDisbursement.transactionNotifier.addObserver(cashAccount.payrollObserver) #cash is observer for payroll disbursements
		payrollDisbursement.transactionNotifier.addObserver(EBPayableAccount.payrollDisbursementObserver)


		#purchase observers

		purchaseProcess.transactionNotifier.addObserver(tradePayablesAccount.purchaseObserver)
		purchaseProcess.transactionNotifier.addObserver(otherExpensesAccount.purchaseObserver)
		purchaseProcess.transactionNotifier.addObserver(prepaidExpensesAccount.purchaseObserver)
		purchaseProcess.transactionNotifier.addObserver(personnelAccount.purchaseObserver)


		#fixed assets
		fixedAssetsProcess.transactionNotifier.addObserver(tradePayablesAccount.fixedAssetsObserver)
		fixedAssetsProcess.transactionNotifier.addObserver(fixedAssetsAccount.fixedAssetsObserver)

		#depreciation
		depreciationProcess.transactionNotifier.addObserver(fixedAssetsAccount.depreciationObserver)
		depreciationProcess.transactionNotifier.addObserver(deprExpenseAccount.depreciationObserver)

		statement = FinancialStatement("Test & Co.", env)
		statement.addAccount(revenueAccount)
		statement.addAccount(tradeReceivablesAccount)
		statement.addAccount(taxPayablesAccount)
		statement.addAccount(cosAccount)
		statement.addAccount(inventoriesAccount)
		statement.addAccount(cashAccount)
		statement.addAccount(EBPayableAccount)
		statement.addAccount(personnelAccount)
		statement.addAccount(otherExpensesAccount)
		statement.addAccount(prepaidExpensesAccount)
		statement.addAccount(fixedAssetsAccount)
		statement.addAccount(deprExpenseAccount)




		env.process(salesHigh.randomTransactions(1000, collections))
		env.process(salesLow.randomTransactions(1000, collections))
		# env.process(manualOrderEvent.start()) #generate random periodic orders
		env.process(collectionsEvent.start()) #payment received 
		env.process(salesTaxDisbur.start()) #collect periodically all the taxes
		env.process(payroll.start()) #start monthly personnel payment
		env.process(payrollDisbursementEvent.start())
		env.process(purchaseProcess.start())
		env.process(disbursementProcess.start())
		env.process(fixedAssetsProcess.start())
		env.process(depreciationProcess.start())

		env.run(until=30)


		return statement


b = BussinessSimulation()
financialStatement = b.simulate()


# misstatement_factions = []
#
# for i in range(50):
# 	b = BussinessSimulation()
# 	financialStatement = b.simulate()
# 	# financialStatement.plotAll()
#
#
# 	misstatement_factions.append(financialStatement.misstatementFraction())
#
#
# plt.hist(misstatement_factions, 50, normed = True)
# x = np.arange(0.001, 0.05, 0.0001)
# alpha, loc, beta = gamma.fit(misstatement_factions)
# plt.plot(x, gamma.pdf(x, alpha, scale=beta))
# plt.show()



# statement.showStatement()

# statement.plotAll()

# [x, y] = statement.historyOfTotalMisstatement()
# [x2, y2, x3, y3] = statement.historyOfTR()



# plt.plot(x,y, '-', x, y2, '-.', x, y3, "--" )
# plt.legend(["Total Misstatement", "TR Correct", "TR Wrong"])
# plt.show()


