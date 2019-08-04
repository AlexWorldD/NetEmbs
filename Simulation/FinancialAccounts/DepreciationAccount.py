# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
DepreciationAccount.py
Created by lex at 2019-03-26.
"""
from Simulation.Abstract.Account import *


class DepreciationAccount(Account):
    def __init__(self, env, name, initialStock=None):
        super().__init__(env, name, initialStock)
        self.depreciationObserver = DepreciationAccount.DepreciationObserver(self)

    def processDepreciation(self, depr):
        depr, fixed_assets, _, _ = depr

        if depr > 0:
            yield self.container.put(depr)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class DepreciationObserver(Observer):
        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processDepreciation(args):
                yield obs
