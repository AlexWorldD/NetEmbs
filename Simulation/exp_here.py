# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
exp_here.py
Created by lex at 2019-03-24.
"""
import simpy
from Simulation.CreateDB import *
from Simulation.Abstract.Transaction import *
from Simulation.BusinessProcesses.Sale import *
from Simulation.BusinessProcesses.Collections import *
from Simulation.BusinessProcesses.GoodsDelivery import *
from Simulation.BusinessProcesses.Purchase import *

connectDB()
cleanDB()
env = simpy.Environment()
sale = SalesTransaction("Sales", 0.2, env)
col = CollectionsTransaction("Collection", env)
delivery = GoodsDeliveryTransaction("Delivery", env)
purchase = PurchaseTransaction("Purchase", env)

for _ in range(10):
    d = sale.newTransaction()
    # col.newTransaction(d)
    # delivery.newTransaction(d)
    purchase.newTransaction()
