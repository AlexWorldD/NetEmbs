# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
finWalk.py
Created by lex at 2019-07-29.
"""

from NetEmbs.GraphSampling.walk_strategies.abstract import abstractWalk
import numpy as np
from scipy.special import softmax
import random
from NetEmbs.FSN.graph import FSN
from NetEmbs.utils.Logs.make_snapshot import log_snapshot
from NetEmbs import CONFIG
from typing import Union, Tuple, Optional, List

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


class metaWalk(abstractWalk):
    """
    Base class for meta-family methods: the ones which follow the direction of relationships rather than edges

    The main difference with the default ones is using a two sub-steps idea
    where the direction for the second sup-step is inversed the direction of initial first sub-step.
    """

    def __init__(self, G: Optional[FSN] = CONFIG.GLOBAL_FSN,
                 walk_length: Optional[int] = CONFIG.WALKS_LENGTH,
                 direction: Optional[str] = "COMBI",
                 pressure: Optional[int] = 10,
                 mode: Optional[int] = 2,
                 allow_back: Optional[bool] = False):
        """
        Build metaWalk class instance
        Parameters
        ----------
        G : FSN object,
            Input graph to be processed
        walk_length : int
            Walk length
        direction : str
            The direction of walk: IN, OUT, COMBI
        pressure : int
            Regularization term for diff function;
                Higher pressure leads to more strict selection,
                while lower, in contrast, tends to ignore the coefficients.
        mode : int
            0 - Uniform, 1 - Weighted, 2 - Difference dependent
        allow_back : bool, default is False
            Allow the walker to move back to the original vertex
        """
        super().__init__(G, walk_length, direction, allow_back)
        self.pressure: int = pressure
        self.mode: int = mode

    def _sub_step_one(self, vertex: Union[str, int]) \
            -> Tuple[Union[str, int], float]:
        """
        The 1st move, from BP to FA
        Parameters
        ----------
        vertex : str or int
                Initial vertex to make a step

        Returns
        -------
        Tuple (node, monetary flow)
            Sampled FA node and monetary flow from initial vertex and that FA
        """
        if not self.G.has_node(vertex):
            raise ValueError(f"Vertex {vertex} is not in FSN!")
        if self.step_direction == "IN":
            candidates = self.G.in_edges(vertex, data=True)
        else:
            candidates = self.G.out_edges(vertex, data=True)
        if len(candidates) > 0:
            try:
                ws = [edge[-1]["weight"] for edge in candidates]
                p_ws = ws / np.sum(ws)
                ins = [edge[["IN", "OUT"].index(self.step_direction)] for edge in candidates]
                if self.mode:
                    tmp_idx = random.choices(range(len(ins)), weights=p_ws, k=1)[0]
                else:
                    tmp_idx = random.choice(range(len(ins)))
                tmp_vertex = ins[tmp_idx]
                tmp_weight = ws[tmp_idx]
                return tmp_vertex, tmp_weight
            except Exception as e:
                snapshot = {"CurrentBPNode": vertex,
                            "NextCandidateFA": list(zip(candidates, ws))}
                log_snapshot(snapshot, __name__)
        else:
            return -1, -1

    def _sub_step_two(self, prev_step: Tuple[Union[str, int], float], original_vertex: Union[str, int]) -> Union[
        str, int]:
        """
        The 2nd move, from FA back to BP set
        Parameters
        ----------
        prev_step : Tuple (node, monetary flow)
            Result of the previous sub-step
        original_vertex : str or int,
            Initial vertex

        Returns
        -------
            Sampled BP nodes w.r.t. the chosen settings
        """
        if prev_step[0] in [-1, -3]:
            return prev_step[0]
        # Instead of following the original direction, we use the opposite one to continue
        if self.mask[self.step_direction] == "IN":
            candidates = self.G.in_edges(prev_step[0], data=True)
        else:
            candidates = self.G.out_edges(prev_step[0], data=True)
        if len(candidates) > 0:
            ws = [edge[-1]["weight"] for edge in candidates]
            outs = [edge[["IN", "OUT"].index(self.mask[self.step_direction])] for edge in candidates]
            if not self.allow_back:
                rm_idx = outs.index(original_vertex)
                ws.pop(rm_idx)
                outs.pop(rm_idx)
                if len(outs) == 0:
                    return -3
            try:
                if self.mode == 2:
                    ws = np.array(ws)
                    ws = diff_function(prev_step[1], ws, self.pressure)
                    res_vertex = random.choices(outs, weights=ws, k=1)[0]
                elif self.mode == 1:
                    ws = ws / np.sum(ws)
                    res_vertex = random.choices(outs, weights=ws, k=1)[0]
                else:
                    res_vertex = random.choice(outs)
                return res_vertex
            except Exception as e:
                snapshot = {"CurrentFANode": prev_step[0],
                            "NextCandidateFA": list(zip(candidates, ws))}
                log_snapshot(snapshot, __name__)

    def step(self, vertex: Union[str, int]) -> Union[str, int]:
        """
        Step method as a sequence of two sub-steps.

        From BP to BP = 1) From BP to FA, and then 2) From FA to BP
        Parameters
        ----------
        vertex : str or int,
            Initial vertex for step

        Returns
        -------
            Sampled vertex from the graph
        """
        return self._sub_step_two(self._sub_step_one(vertex), vertex)

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


class metaUniform(metaWalk):
    """
    metaUniform walking strategy: all transition probabilities are uniform
    """
    def __init__(self, G: Optional[FSN] = CONFIG.GLOBAL_FSN,
                 walk_length: Optional[int] = CONFIG.WALKS_LENGTH,
                 direction: Optional[str] = "COMBI",
                 allow_back: Optional[bool] = False, **kwargs):
        """

        Parameters
        ----------
        G : FSN object,
            Input graph to be processed
        walk_length : int
            Walk length
        direction : str
            The direction of walk: IN, OUT, COMBI
        allow_back : bool, default is False
            Allow the walker to move back to the original vertex
        """
        # Wrapper around the metaWalk, which default values represent the Uniform behaviour
        super().__init__(G, walk_length, direction, None, 0, allow_back)


class metaWeighted(metaWalk):
    """
        metaWeighted walking strategy: use weighted transition probabilites
    """
    def __init__(self, G: Optional[FSN] = CONFIG.GLOBAL_FSN,
                 walk_length: Optional[int] = CONFIG.WALKS_LENGTH,
                 direction: Optional[str] = "COMBI",
                 allow_back: Optional[bool] = False, **kwargs):
        """

        Parameters
        ----------
        G : FSN object,
            Input graph to be processed
        walk_length : int
            Walk length
        direction : str
            The direction of walk: IN, OUT, COMBI
        allow_back : bool, default is False
            Allow the walker to move back to the original vertex
        """
        # Wrapper around the metaWalk, which default values represent the Weighted behaviour
        super().__init__(G, walk_length, direction, None, 1, allow_back)


class finWalk(metaWalk):
    """
    finWalk implementation

    finWalk method use weighted transition probability for the first move (from BP to FA)
    and the difference dependent transition probability for the second move (back to BP)
    """

    def __init__(self, G: Optional[FSN] = CONFIG.GLOBAL_FSN,
                 walk_length: Optional[int] = CONFIG.WALKS_LENGTH,
                 direction: Optional[str] = "COMBI",
                 pressure: Optional[int] = 10,
                 allow_back: Optional[bool] = False, **kwargs):
        """

        Parameters
        ----------
        G : FSN object,
            Input graph to be processed
        walk_length : int
            Walk length
        direction : str
            The direction of walk: IN, OUT, COMBI
        pressure : int
            Regularization term for diff function;
                Higher pressure leads to more strict selection,
                while lower, in contrast, tends to ignore the coefficients.
        allow_back : bool, default is False
            Allow the walker to move back to the original vertex
        """
        # Wrapper around the metaWalk, which default values represent the finWalk behaviour
        super().__init__(G, walk_length, direction, pressure, 2, allow_back)
