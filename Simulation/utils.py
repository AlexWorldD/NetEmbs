# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
utils.py
Created by lex at 2019-03-24.
"""

import random
import string


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))
