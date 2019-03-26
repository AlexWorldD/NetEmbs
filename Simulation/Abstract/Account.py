# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
Account.py
Created by lex at 2019-03-24.
"""
from Abstract.Observer import *
import simpy


class Account(Observable):
    def __init__(self, env, name, initial_stock=None):
        """
        :param env: Environment of simpy simulation
        :param name: Unique FA name
        :param fa_name: Group name of FA such as Revenue, Tax etc.
        :param initial_stock: initial value of stocks for current FA
        """
        Observable.__init__(self)

        print("Initialize container with name: ", name)
        if initial_stock is not None:
            self.container = simpy.Container(env, init=initial_stock)
        else:
            self.container = simpy.Container(env)
        self.env = env
        self.name = name

    def __str__(self):
        return "Account name: %s \t\t level correct: %d wrong: %d" % (self.name, self.container.level)

    def __repr__(self):
        return "%s" % self.name
