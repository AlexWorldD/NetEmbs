from Observer import * 
import matplotlib.pyplot as plt

#General ledger

#snapshot of all accounts after each update
#snapshot of all ratios after each update

class FinancialStatement(object):

	def __init__(self, name, env):
		self.name 		= name
		self.env 		= env
		self.accounts 	= []
		self.totalMisstatement = []
		self.snapshotMoment = []
		print "Financial statement of %s" % name
		self.accountObserver = FinancialStatement.AccountChangeObserver(self)


		self.snapshotAllAccountsCorrect = {'value' : []}
		self.snapshotAllAccountsWrong = {'value' : []}


	def misstatementFraction(self):

		y_c = self.snapshotAllAccountsCorrect['value']
		y_w = self.snapshotAllAccountsWrong['value']

		y_t = y_c[len(y_c) - 1] + y_w[len(y_w) - 1]
		y_frac = 1.0*y_w[len(y_w) - 1]/y_t

		return y_frac


	def addAccount(self, account):
		self.accounts.append(account)
		account.addObserver(self.accountObserver) #add self as observer for account change
		self.snapshotAllAccountsCorrect[account.name] = []
		self.snapshotAllAccountsWrong[account.name] = []

	def showStatement(self):
		totalError = 0
		for account in self.accounts:
			totalError += account.containerWrong.level
			print "Account name: %s \n \t\t value:%d \t\t(incorrect:%d)" % (account.name, account.containerCorrect.level, account.containerWrong.level)
		print "Total misstatement in financial report:\t %d" % totalError


	def snapshotAll(self):
		total_c = 0
		total_w = 0
		for account in self.accounts:
			self.snapshotAllAccountsCorrect[account.name].append(account.containerCorrect.level)
			self.snapshotAllAccountsWrong[account.name].append(account.containerWrong.level)
			total_c += account.containerCorrect.level
			total_w += account.containerWrong.level

		self.snapshotAllAccountsWrong['value'].append(total_w)
		self.snapshotAllAccountsCorrect['value'].append(total_c)

		self.snapshotMoment.append(self.env.now)


	def plotAll(self):
		i = 1
		for account in self.accounts:
			y_c = self.snapshotAllAccountsCorrect[account.name]
			y_w = self.snapshotAllAccountsWrong[account.name]
			plt.figure(i)
			plt.plot(self.snapshotMoment, y_c, '-', self.snapshotMoment, y_w, '--')
			plt.legend(["Correct", "Wrong"])
			plt.title(account.name)
			i += 1

		y_c = self.snapshotAllAccountsCorrect['value']
		y_w = self.snapshotAllAccountsWrong['value']

		y_t = [a + b for a, b in zip(y_c, y_w)]


		plt.figure(i)
		plt.plot(self.snapshotMoment, y_t, '-', self.snapshotMoment, y_w, '--')
		plt.legend(['Total monetary value', 'Misstatement'])
		plt.title('Misstatement as part of the total')

		y_frac = [1.0*a/b for a, b in zip(y_w, y_t)]

		plt.figure(i+1)
		plt.plot(self.snapshotMoment, y_frac)
		plt.legend(['Misstatement fraction'])
		plt.title('Misstatement as a fraction of the total')

		plt.show()



	class AccountChangeObserver(Observer):

		def __init__(self, outer):
			self.outer = outer

		def update(self, observable, args):

			self.outer.snapshotAll()
			yield self.outer.env.timeout(0)