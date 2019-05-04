# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
cleaning_exp.py
Created by lex at 2019-05-04.
"""
from NetEmbs.DataProcessing import *
from NetEmbs.GenerateData.complex_df import zerosData, dirtyData


if __name__ == '__main__':
    zer = zerosData()
    p_zer = prepare_data(zer, split=False)
    prepare_data("t")