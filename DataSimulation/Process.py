from Observer import * 

class Process(object):
	class TransactionNotifier(Observable):
		def __init__(self, outer):
			Observable.__init__(self)
			self.outer = outer

		def notifyObservers(self, args = None):
			self.setChanged()
			for obs in Observable.notifyObservers(self, args):
				yield obs

	class TransactionNotifee(Observer):
		def __init__(self, outer):
			self.outer = outer

		def update(self, observable, args):
			print "Received a notification from:%s" % observable.outer.name