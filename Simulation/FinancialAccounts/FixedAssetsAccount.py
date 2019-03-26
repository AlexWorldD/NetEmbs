# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
FixedAssetsAccount.py
Created by lex at 2019-03-26.
"""
from Simulation.Abstract.Account import *


class FixedAssetsAccount(Account):

    def __init__(self, env, name, initialStock=None):
        super().__init__(env, name, initialStock)
        self.fixedAssetsObserver = FixedAssetsAccount.FixedAssetsObserver(self)
        self.depreciationObserver = FixedAssetsAccount.DepreciationObserver(self)

    def processDepreciation(self, depr):
        depr, fixed_assets = depr

        if fixed_assets > 0:
            yield self.container.get(fixed_assets)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class DepreciationObserver(Observer):
        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processDepreciation(args):
                yield obs

    def processFixedAssets(self, assets):
        trade_pay, fixed_assets = assets

        if fixed_assets > 0:
            yield self.container.put(fix)

        self.setChanged()

        for obs in self.notifyObservers():
            yield obs

    class FixedAssetsObserver(Observer):
        def __init__(self, outer):
            self.outer = outer

        def update(self, observable, args):
            for obs in self.outer.processFixedAssets(args):
                yield obs
