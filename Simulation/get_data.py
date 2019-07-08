# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
get_data.py
Created by lex at 2019-03-26.
"""
from Simulation.CreateDB import *
from Simulation.FSN_Simulation import FSN_Simulation


cleanDB(db_file="FSN_Data_5k.db")
b = FSN_Simulation()
financialStatement = b.simulate(SalesNum=(500, 500), until=2500)
