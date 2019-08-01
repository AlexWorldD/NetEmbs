# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
randomWalk.py
Created by lex at 2019-07-31.
"""
from NetEmbs.GraphSampling.walk_strategies.abstract import abstractWalk
import numpy as np
import random
from NetEmbs.FSN.graph import FSN
from NetEmbs.utils.Logs.make_snapshot import log_snapshot
from NetEmbs import CONFIG
from typing import Union, Optional, List

np.seterr(all="raise")


class originalRandomWalk(abstractWalk):
    def __init__(self, G: Optional[FSN] = CONFIG.GLOBAL_FSN,
                 walk_length: Optional[int] = CONFIG.WALKS_LENGTH,
                 allow_back: Optional[bool] = False, **kwargs):
        super().__init__(G, walk_length, "RANDOM", allow_back)
        self.G: FSN = G

    def _sub_step(self, vertex: Union[str, int]) -> Union[str, int]:
        if not self.G.has_node(vertex):
            raise ValueError(f"Vertex {vertex} is not in FSN!")
        candidates = (self.G.in_edges(vertex, data=True), self.G.out_edges(vertex, data=True))
        candidates = list(zip([edge[0] for edge in candidates[0]], [edge[-1]["weight"] for edge in candidates[0]])) + \
                     list(zip([edge[1] for edge in candidates[1]], [edge[-1]["weight"] for edge in candidates[1]]))
        if len(candidates) > 0:
            try:
                candidates = [edge[0] for edge in candidates]
                tmp_vertex = random.choice(candidates)
                return tmp_vertex
            except Exception as e:
                snapshot = {"CurrentBPNode": vertex,
                            "NextCandidateFA": candidates}
                log_snapshot(snapshot, __name__, e)
        else:
            return -1

    def step(self, vertex: Union[str, int]) -> Union[str, int]:
        return self._sub_step(self._sub_step(vertex))

    def walk(self, vertex: Union[str, int]) -> List[Union[str, int]]:
        context = list()
        context.append(vertex)
        while len(context) < self.walk_length:
            new_node = self.step(context[-1])
            if new_node not in [-1, -3]:
                context.append(new_node)
            else:
                print("Cannot continue walking... Termination.")
                break
        return context
