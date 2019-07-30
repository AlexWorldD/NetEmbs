# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
abstract.py
Created by lex at 2019-07-30.
"""
import numpy as np
from NetEmbs.FSN.graph import FSN
from typing import Optional, Union, Dict

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
        self.walk_direction: str = direction
        self.allow_back: bool = allow_back
        self.step_direction: str = self.walk_direction if self.walk_direction in ["IN", "OUT", "RANDOM"] else "IN"
        self.mask: Dict[str, str] = {"IN": "OUT", "OUT": "IN"}

    def step(self, vertex: Union[str, int]) -> Union[str, int]:
        pass
