# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
batch_generator.py
Created by lex at 2019-08-01.
"""
import numpy as np


def generate_batch(all_data, batch_size):
    _t = np.random.randint(0, len(all_data), batch_size)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32, buffer=np.array([all_data[t][0] for t in _t]))
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32, buffer=np.array([all_data[t][1] for t in _t]))
    return batch, context
