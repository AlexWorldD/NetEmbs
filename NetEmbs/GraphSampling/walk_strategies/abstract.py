# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
abstract.py
Created by lex at 2019-07-30.
"""
import numpy as np
from NetEmbs.FSN.graph import FSN
from typing import Optional, Union

np.seterr(all="raise")


class abstractWalk():
    """
    Abstract base class for any walking strategy
    """

    def __init__(self, G: Optional[FSN],
                 walk_length: Optional[int],
                 direction: Optional[str],
                 allow_back: Optional[bool] = False):
        self.G: FSN = G
        self.walk_length: int = walk_length
        self.direction: str = direction
        self.allow_back: bool = allow_back
        self.move_in: bool = True if self.direction in ["IN", "COMBI"] else False

    def step(self, vertex: Union[str, int]) -> Union[str, int]:
        pass
