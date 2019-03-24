# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
exp_here.py
Created by lex at 2019-03-24.
"""

from Simulation.CreateDB import *
from Simulation.Transaction import *
connectDB()
tr = Transaction(1, 2)
tr.addRecord("TestAAA", "Test", 100, 1)