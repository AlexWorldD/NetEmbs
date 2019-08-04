# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
get_data.py
Created by lex at 2019-03-26.
"""
import simpy
from Simulation.CreateDB import *
from Simulation.FSN_Simulation import FSN_Simulation

cleanDB()
b = FSN_Simulation()
financialStatement = b.simulate()
