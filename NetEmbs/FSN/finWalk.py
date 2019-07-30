# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
finWalk.py
Created by lex at 2019-07-29.
"""

import numpy as np
import random
from scipy.special import softmax
import random
import networkx as nx
from networkx.algorithms import bipartite
from collections import Counter
import pandas as pd
from NetEmbs.FSN.graph import FSN
from NetEmbs.utils.Logs.make_snapshot import log_snapshot
import logging
from NetEmbs.CONFIG import LOG, all_sampling_strategies, N_JOBS
from NetEmbs import CONFIG
import time
from pathos.multiprocessing import ProcessPool
import itertools
import os
from NetEmbs.utils.get_size import get_size
from tqdm.auto import tqdm
import pickle
from typing import Union, Tuple, Optional

np.seterr(all="raise")


def diff_function(prev_edge, new_edges, pressure):
    """
    Function for calculation transition probabilities based on Differences between previous edge and candidate edge
    :param prev_edge: Monetary amount on previous edge
    :param new_edges: Monetary amount on all edges candidates
    :param pressure: The regularization term, higher pressure leads to more strict function
    :return: array of transition probabilities
    """
    return softmax((1.0 - abs(new_edges - prev_edge)) * pressure)


def sub_step_one(G: FSN, vertex: Union[str, int], direction: str = 'IN', weighted: bool = True) \
        -> Tuple[Union[str, int], float]:
    if not G.has_node(vertex):
        raise ValueError(f"Vertex {vertex} is not in FSN!")
    if direction == "IN":
        candidates = G.in_edges(vertex, data=True)
    elif direction == "OUT":
        candidates = G.out_edges(vertex, data=True)
    else:
        raise ValueError(f"Wrong direction argument! {direction} used while IN or OUT are allowed!")
    if len(candidates) > 0:
        try:
            ws = [edge[-1]["weight"] for edge in candidates]
            p_ws = ws / np.sum(ws)
            ins = [edge[["IN", "OUT"].index(direction)] for edge in candidates]
            if weighted:
                tmp_idx = random.choices(range(len(ins)), weights=p_ws, k=1)[0]
            else:
                tmp_idx = random.choice(range(len(ins)))
            tmp_vertex = ins[tmp_idx]
            tmp_weight = ws[tmp_idx]
            return (tmp_vertex, tmp_weight)
        except Exception as e:
            snapshot = {"CurrentBPNode": vertex,
                        "NextCandidateFA": list(zip(candidates, ws))}
            log_snapshot(snapshot, __name__, e)
    else:
        return (-1, -1)


def sub_step_two(G: FSN, prev_step: Tuple[Union[str, int], float], original_vertex: Union[str, int],
                 direction: str = 'IN', mode: int = 2, pressure: Optional[float] = 1.0) -> Union[str, int]:
    if prev_step[0] == -1:
        return -1
    if direction == "IN":
        candidates = G.in_edges(prev_step[0], data=True)
    elif direction == "OUT":
        candidates = G.out_edges(prev_step[0], data=True)
    else:
        raise ValueError(f"Wrong direction argument! {direction} used while IN or OUT are allowed!")
    if len(candidates) > 0:
        ws = [edge[-1]["weight"] for edge in candidates]
        outs = [edge[["IN", "OUT"].index(direction)] for edge in candidates]
        rm_idx = outs.index(original_vertex)
        ws.pop(rm_idx)
        outs.pop(rm_idx)
        if len(outs) == 0:
            return -3
        try:
            if mode == 2:
                ws = diff_function(prev_step[1], ws, pressure)
                res_vertex = random.choices(outs, weights=ws, k=1)[0]
            elif mode == 1:
                ws = ws / np.sum(ws)
                res_vertex = random.choices(outs, weights=ws, k=1)[0]
            else:
                res_vertex = random.choice(outs)
            return res_vertex
        except Exception as e:
            snapshot = {"CurrentFANode": prev_step[0],
                        "NextCandidateFA": list(zip(candidates, ws))}
            log_snapshot(snapshot, __name__, e)


mask = {"IN": "OUT", "OUT": "IN"}


def step(G: FSN, vertex: Union[str, int], direction: str = 'IN', pressure: Optional[float] = 1.0):
    return sub_step_two(G, sub_step_one(G, vertex, direction), vertex, mask[direction], pressure=pressure)
