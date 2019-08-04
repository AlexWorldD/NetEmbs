# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
construct_grid_params.py
Created by lex at 2019-07-08.
"""
from itertools import product


def get_GRID(params):
    """
    Construct the grid over the given parameters values
    :param params:
    :return: List of dictionaries to be used for one iteration of End2End execution
    """
    parameter_values = list(product(*params.values()))
    parameter_keys = [key.upper() for key in params.keys()]
    return [dict(list(zip(parameter_keys, values))) for values in parameter_values]
