from Process import *
from Transaction import *

class PeriodAllocation(Process):

	def __init__(self, env, name, OtherExpensesAccount, PrepaidExpensesAccount, AccrualsAccount, personnelAccount):
		self.name = name
		self.env = env
		self._otherExpensesAccount = OtherExpensesAccount
		self._prepaidExpensesAccount = PrepaidExpensesAccount
		self._accrualsAccount = AccrualsAccount
		self._personnelAccount = personnelAccount