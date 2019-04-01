# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
complex_df.py
Created by lex at 2019-03-14.
"""
import pandas as pd
import random


# Construct test case with Sale BPs with different tax rates
def sales_collections(N=2, taxes=[0.06, 0.25], noise=[1], titles=["ID", "Name", "Journal", "Date", "Debit", "Credit"]):
    if taxes is not None:
        tax_rates = taxes
    if N is not None:
        N = N
    cur_id = 0
    data = pd.DataFrame(columns=titles)
    for tax in tax_rates:
        for _ in range(N):
            # generate amounts
            rev = random.randint(10, 1000)
            if 1 in noise:
                t = rev * tax + random.randint(2, 10)
            else:
                t = rev * tax
            if not 2 in noise:
                tr = rev + t
                data = data.append(
                    [pd.Series([cur_id, "Revenue", "Sales ledger", "01/01/2017", 0.0, rev], index=data.columns),
                     pd.Series([cur_id, "Tax", "Sales ledger", "01/01/2017", 0.0, t], index=data.columns),
                     pd.Series([cur_id, "Trade Receivables", "Sales ledger", "01/01/2017", tr, 0.0],
                               index=data.columns)]
                    , ignore_index=True)
                cur_id += 1
                data = data.append([pd.Series([cur_id, "Trade Receivables", "Sales ledger", "01/01/2017", 0.0, tr],
                                              index=data.columns),
                                    pd.Series([cur_id, "Cash", "Sales ledger", "01/01/2017", tr, 0.0],
                                              index=data.columns)]
                                   , ignore_index=True)
                cur_id += 1
            else:
                left_noise = random.randint(5, int(rev) / 20)
                right_noise = random.randint(5, int(rev) / 20)
                tr = rev + t + left_noise - right_noise
                data = data.append(
                    [pd.Series([cur_id, "Revenue", "Sales ledger", "01/01/2017", 0.0, rev], index=data.columns),
                     pd.Series([cur_id, "Tax", "Sales ledger", "01/01/2017", 0.0, t], index=data.columns),
                     pd.Series([cur_id, "LeftNoise", "Sales ledger", "01/01/2017", 0.0, left_noise],
                               index=data.columns),
                     pd.Series([cur_id, "Trade Receivables", "Sales ledger", "01/01/2017", tr, 0.0],
                               index=data.columns),
                     pd.Series([cur_id, "RightNoise", "Sales ledger", "01/01/2017", right_noise, 0.0],
                               index=data.columns)]
                    , ignore_index=True)
                cur_id += 1
                data = data.append([pd.Series([cur_id, "Trade Receivables", "Sales ledger", "01/01/2017", 0.0, tr],
                                              index=data.columns),
                                    pd.Series([cur_id, "Cash", "Sales ledger", "01/01/2017", tr, 0.0],
                                              index=data.columns)]
                                   , ignore_index=True)
                cur_id += 1
    return data
