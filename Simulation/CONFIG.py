# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
CONFIG.py
Created by lex at 2019-03-24.
"""
import numpy as np

DB_PATH = "FSN_Data.db"
PRINT = False
VARIANTS = list(range(13))
TRANSACTIONS_LIMITS = {"Sales": (10, 1000), "Depreciation": (10, 100), "FixedAssets": (10, 100), "Purchase": (1, 100)}
# Transaction Noises
NOISE = {"Sales": True, "Collections": False, "GoodsDelivery": True, "Depreciation": True,
         "FixedAssets": True, "Purchase": True}
NO_NOISE = False
if NO_NOISE:
    NOISE = {"Sales": False, "Collections": False, "GoodsDelivery": False, "Depreciation": False,
             "FixedAssets": False, "Purchase": False}
NOISE_Type1 = {"freq": 0.9, "amplitude": 0.01}
NOISE_Type2 = {"freq": 0.5, "proportion": 0.5, "num_amplitude": 5.0, "noise_amplitude": 0.01}
