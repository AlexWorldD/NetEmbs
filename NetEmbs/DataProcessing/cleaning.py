# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
cleaning.py
Created by lex at 2019-05-04.
"""


# Helper function to understand the source of errors
def countStrings(df, col=["Credit", "Debit"]):
    output = dict()
    for title in col:
        output[title] = df[title].map(lambda x: 1 if type(x) == str else 0).sum()
    return output


def countNaN(df, col=["Credit", "Debit"]):
    output = dict()
    for title in col:
        output[title] = df[title].isnull().sum()
    return output


def countDirtyData(df, col=["Credit", "Debit"]):
    print("Strings in numeric columns: ", countStrings(df, col))
    print("NaN in numeric columns: ", countNaN(df, col))


def CreditDebit(row):
    try:
        row["Credit"] = abs(row["Value"]) if row["type"] == "credit" else 0.0
        row["Debit"] = abs(row["Value"]) if row["type"] == "debit" else 0.0
    except Exception:
        raise KeyError("Cannot find 'Value' column in given DataFrame!")
    return row


def delStrings(df, col_names=["Value"]):
    for title in col_names:
        df[title] = df[title].map(lambda x: x if type(x) is not str else None)
    return df.dropna(subset=col_names)
