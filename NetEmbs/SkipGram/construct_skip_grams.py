# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
construct_skip_grams.py
Created by lex at 2019-08-01.
"""
from NetEmbs.GraphSampling.walk_strategies import *
import numpy as np
import random
from NetEmbs.FSN.graph import FSN
from NetEmbs.utils.Logs.make_snapshot import log_snapshot
from NetEmbs import CONFIG
from typing import Union, Tuple, Optional, List
import os
import time
from pathos.multiprocessing import ProcessPool
import itertools
import os
import logging
import pickle
from NetEmbs.utils.get_size import get_size
from tqdm.auto import tqdm

def get_SkipGrams()