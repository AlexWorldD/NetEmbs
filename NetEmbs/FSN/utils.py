# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
utils.py
Created by lex at 2019-03-15.
"""
import numpy as np
from scipy.special import softmax


def def_step(G, vertex, direction="IN", weighted=False, debug=False, **kwargs):
    """
     One step according to the original implementation of RandomWalk by Perozzi et al.
    :param G: graph/network on which step should be done
    :param vertex: current vertex
    :param direction: the direction of step: IN or OUT
    :param weighted: use the edge's weight for transition probability
    :param debug: print intermediate stages
    :param kwargs:
    :return: next step if succeeded or -1 if failed
    """
    if vertex == -1:
        #         Step cannot be made, return -1
        return -1
    elif not G.has_node(vertex):
        raise ValueError("Vertex {!r} is not in FSN!".format(vertex))
    if direction == "IN":
        ins = G.in_edges(vertex, data=True)
    elif direction == "OUT":
        ins = G.out_edges(vertex, data=True)
    else:
        raise ValueError("Wrong direction argument! {!s} used while IN or OUT are allowed!".format(direction))
    indexes = ["IN", "OUT"]
    if len(ins) > 0:
        if weighted:
            ws = [edge[-1]["weight"] for edge in ins]
            p_ws = ws / np.sum(ws)
            ins = [edge[indexes.index(direction)] for edge in ins]
            tmp_idx = np.random.choice(range(len(ins)), p=p_ws)
            tmp_vertex = ins[tmp_idx]
            tmp_weight = ws[tmp_idx]
        else:
            ins = [edge[indexes.index(direction)] for edge in ins]
            tmp_vertex = np.random.choice(ins)
        if debug:
            print(tmp_vertex)
        return tmp_vertex
    else:
        return -1


def step(G, vertex, direction="IN", mode=2, allow_back=False, pressure=20, debug=False):
    # ////// THE FIRST STEP TO OPPOSITE SET OF NODES \\\\\
    if vertex == -1:
        #         Step cannot be made, return -1
        return -1
    elif not G.has_node(vertex):
        raise ValueError("Vertex {!r} is not in FSN!".format(vertex))
    if direction == "IN":
        ins = G.in_edges(vertex, data=True)
    elif direction == "OUT":
        ins = G.out_edges(vertex, data=True)
    else:
        raise ValueError("Wrong direction argument! {!s} used while IN or OUT are allowed!".format(direction))
    mask = {"IN": "OUT", "OUT": "IN"}
    indexes = ["IN", "OUT"]
    if len(ins) > 0:
        ws = [edge[-1]["weight"] for edge in ins]
        p_ws = ws / np.sum(ws)
        ins = [edge[indexes.index(direction)] for edge in ins]
        if mode == 0:
            tmp_idx = np.random.choice(range(len(ins)))
        else:
            tmp_idx = np.random.choice(range(len(ins)), p=p_ws)
        tmp_vertex = ins[tmp_idx]
        tmp_weight = ws[tmp_idx]
        if debug:
            print(tmp_vertex)
    else:
        return -1
    # ////// THE SECOND STEP TO OPPOSITE SET OF NODES (to original one) \\\\\
    if mask[direction] == "IN":
        outs = G.in_edges(tmp_vertex, data=True)
    elif mask[direction] == "OUT":
        outs = G.out_edges(tmp_vertex, data=True)
    if len(outs) > 0:
        ws = [edge[-1]["weight"] for edge in outs]
        outs = [edge[indexes.index(mask[direction])] for edge in outs]
        if not allow_back:
            rm_idx = outs.index(vertex)
            ws.pop(rm_idx)
            outs.pop(rm_idx)
        if len(outs) == 0:
            return -3
        ws = np.array(ws)
        if mode == 2:
            # Transition probability depends on the difference between monetary flows
            ws = softmax((1.0 - abs(ws - tmp_weight)) * pressure)
        if mode == 1:
            # Transition probability depends on the monetary flows - "rich gets richer"
            ws = ws / np.sum(ws)
        elif mode == 0:
            # Transition probability is uniform
            if debug:
                print(outs)
            return np.random.choice(outs)
        if debug:
            print(list(zip(outs, ws)))
        return np.random.choice(outs, p=ws)
    else:
        return -2
