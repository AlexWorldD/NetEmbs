# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
__init__.py.py
Last modified by lex at 2019-03-14.
"""
from NetEmbs.FSN.graph import FSN
from NetEmbs.FSN.utils import default_step, step
from NetEmbs.FSN.utils import randomWalk, make_pairs
from NetEmbs.FSN.utils import get_SkipGrams, get_SkipGrams_raw
from NetEmbs.FSN.utils import TransformationBPs

from NetEmbs.FSN.finWalk import sub_step_one, sub_step_two, step as new_step
