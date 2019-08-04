# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
defaultWalk.py
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


class defWalk(abstractWalk):
    def __init__(self, G: Optional[FSN] = CONFIG.GLOBAL_FSN,
                 walk_length: Optional[int] = CONFIG.WALKS_LENGTH,
                 direction: Optional[str] = "COMBI",
                 mode: Optional[int] = 1,
                 allow_back: Optional[bool] = False):
        super().__init__(G, walk_length, direction, allow_back)
        self.mode: int = mode

    def _sub_step(self, vertex: Union[str, int]) -> Union[str, int]:
        if not self.G.has_node(vertex):
            raise ValueError(f"Vertex {vertex} is not in FSN!")
        if self.step_direction == "IN":
            candidates = self.G.in_edges(vertex, data=True)
            candidates = list(zip([edge[0] for edge in candidates], [edge[-1]["weight"] for edge in candidates]))
        else:
            candidates = self.G.out_edges(vertex, data=True)
            list(zip([edge[1] for edge in candidates], [edge[-1]["weight"] for edge in candidates]))
        if len(candidates) > 0:
            try:
                if self.mode:
                    ws = [edge[-1] for edge in candidates]
                    p_ws = ws / np.sum(ws)
                    candidates = [edge[0] for edge in candidates]
                    tmp_idx = random.choices(range(len(candidates)), weights=p_ws, k=1)[0]
                    tmp_vertex = candidates[tmp_idx]
                else:
                    candidates = [edge[0] for edge in candidates]
                    tmp_vertex = random.choice(candidates)
                return tmp_vertex
            except Exception as e:
                snapshot = {"CurrentBPNode": vertex,
                            "NextCandidateFA": candidates}
                log_snapshot(snapshot, __name__)
        else:
            return -1

    def step(self, vertex: Union[str, int]) -> Union[str, int]:
        return self._sub_step(self._sub_step(vertex))

    def walk(self, vertex: Union[str, int]) -> List[Union[str, int]]:
        context = list()
        context.append(vertex)
        while len(context) < self.walk_length:
            new_node = self.step(context[-1])
            if self.walk_direction == "COMBI":
                self.step_direction = self.mask[self.step_direction]
            if new_node not in [-1, -3]:
                context.append(new_node)
            elif self.walk_direction != "COMBI":
                print("Cannot continue walking... Termination.")
                break
        return context


class defUniform(defWalk):
    def __init__(self, G: Optional[FSN] = CONFIG.GLOBAL_FSN,
                 walk_length: Optional[int] = CONFIG.WALKS_LENGTH,
                 direction: Optional[str] = "COMBI",
                 allow_back: Optional[bool] = False, **kwargs):
        super().__init__(G, walk_length, direction, 0, allow_back)


class defWeighted(defWalk):
    def __init__(self, G: Optional[FSN] = CONFIG.GLOBAL_FSN,
                 walk_length: Optional[int] = CONFIG.WALKS_LENGTH,
                 direction: Optional[str] = "COMBI",
                 allow_back: Optional[bool] = False, **kwargs):
        super().__init__(G, walk_length, direction, 1, allow_back)
