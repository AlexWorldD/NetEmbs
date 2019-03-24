# encoding: utf-8
__author__ = 'Marcel Boersma'
"""
Observer.py
Created by Marcel
"""


class Observer(object):
    def update(observable, arg):
        pass


class Observable(object):
    def __init__(self):
        self.obs = []
        self.changed = 0

    def addObserver(self, observer):
        if observer not in self.obs:
            self.obs.append(observer)

    def deleteObserver(self, observer):
        if observer in self.obs:
            self.obs.remove(observer)

    def notifyObservers(self, arg=None):
        if not self.changed: return

        localArray = self.obs[:]
        self.clearChanged()

        for observer in localArray:
            for obs in observer.update(self, arg):
                yield obs

    def deleteObservers(self):
        self.obs = []

    def setChanged(self):
        self.changed = 1

    def clearChanged(self):
        self.changed = 0

    def hasChanged(self):
        return self.changed

    def countObservers(self):
        return len(self.obs)
