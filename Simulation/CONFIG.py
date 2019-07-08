# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
CONFIG.py
Created by lex at 2019-03-24.
"""
import numpy as np

DB_PATH = "FSN_Data_5k.db"
PRINT = False
VARIANTS = list(range(13))
TRANSACTIONS_LIMITS = {"Sales": (10, 1000), "Depreciation": (10, 100), "FixedAssets": (10, 100), "Purchase": (1, 100)}
# Transaction Noises
NOISE = {"Sales": True, "Collections": False, "GoodsDelivery": True, "Depreciation": True,
         "FixedAssets": True, "Purchase": True, "Payroll": True}
NO_NOISE = False
if NO_NOISE:
    NOISE = {"Sales": False, "Collections": False, "GoodsDelivery": False, "Depreciation": False,
             "FixedAssets": False, "Purchase": False, "Payroll": False}
NOISE_Type1 = {"freq": 0.9, "amplitude": 0.01}
NOISE_Type2 = {"freq": 0.5, "proportion": 0.5, "num_amplitude": 5.0, "noise_amplitude": 0.005}

# Noisiness as in RealData
real_left = {1: 0.2455795677799607,
             2: 0.137524557956778,
             3: 0.06286836935166994,
             4: 0.07072691552062868,
             5: 0.08644400785854617,
             6: 0.11787819253438114,
             7: 0.137524557956778,
             8: 0.09823182711198428,
             9: 0.03929273084479371,
             10: 0.003929273084479371}

ks_l, pds_l = [0, 0] + list(range(1, 9)), list(real_left.values())

noisy_left = list()

real_right = {1: 0.42857142857142855,
              2: 0.16071428571428573,
              3: 0.039285714285714285,
              4: 0.04642857142857143,
              5: 0.060714285714285714,
              6: 0.08392857142857142,
              7: 0.08928571428571429,
              8: 0.07142857142857142,
              9: 0.017857142857142856,
              10: 0.0017857142857142857}
ks_r, pds_r = [0] + list(range(1, 10)), list(real_right.values())

noisy_right = list()
