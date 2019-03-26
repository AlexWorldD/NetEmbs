# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
utils.py
Created by lex at 2019-03-15.
"""
import numpy as np
from scipy.special import softmax
import random
import networkx as nx
from networkx.algorithms import bipartite
from NetEmbs.CONFIG import *


def default_step(G, vertex, direction="IN", mode=0, return_full_step=False, debug=False):
    """
     One step according to the original implementation of RandomWalk by Perozzi et al.
     (uniform probabilities, follows the same direction)
    :param G: graph/network on which step should be done
    :param vertex: current vertex
    :param direction: the direction of step: IN or OUT
    :param mode: use the edge's weight for transition probability
    :param return_full_step: if True, then step includes intermediate node of FA type
    :param debug: print intermediate stages
    :return: next step if succeeded or -1 if failed
    """
    if vertex in [-1, -2, -3]:
        #         Step cannot be made, return -1
        return vertex
    elif not G.has_node(vertex):
        raise ValueError("Vertex {!r} is not in FSN!".format(vertex))
    if mode in [0, 1]:
        pass
    else:
        raise ValueError(
            "For DefaultStep only two modes available: 0 (uniform) or 1(weighted) byt given {!r}!".format(mode))
    # Get the neighborhood of current node regard the chosen direction
    if direction == "IN":
        ins = G.in_edges(vertex, data=True)
    elif direction == "OUT":
        ins = G.out_edges(vertex, data=True)
    else:
        raise ValueError("Wrong direction argument! {!s} used while IN or OUT are allowed!".format(direction))
    output = list()
    indexes = ["IN", "OUT"]
    # Check that we can make step, otherwise return special value -1
    if len(ins) > 0:
        # Apply weighted probabilities
        if mode == 1:
            ws = [edge[-1]["weight"] for edge in ins]
            p_ws = ws / np.sum(ws)
            ins = [edge[indexes.index(direction)] for edge in ins]
            tmp_idx = np.random.choice(range(len(ins)), p=p_ws)
            tmp_vertex = ins[tmp_idx]
            tmp_weight = ws[tmp_idx]
        #     Apply uniform probabilities
        elif mode == 0:
            ins = [edge[indexes.index(direction)] for edge in ins]
            tmp_vertex = np.random.choice(ins)
        if debug:
            print(tmp_vertex)
        output.append(tmp_vertex)
    else:
        return -1
    # ///////////// \\\\\\\\\\\\\\\
    #     Second sub-step, from FA to BP
    if direction == "IN":
        ins = G.in_edges(tmp_vertex, data=True)
    elif direction == "OUT":
        ins = G.out_edges(tmp_vertex, data=True)
    else:
        raise ValueError("Wrong direction argument! {!s} used while IN or OUT are allowed!".format(direction))
    # Check that we can make step, otherwise return special value -1
    if len(ins) > 0:
        if mode == 1:
            ws = [edge[-1]["weight"] for edge in ins]
            p_ws = ws / np.sum(ws)
            ins = [edge[indexes.index(direction)] for edge in ins]
            tmp_idx = np.random.choice(range(len(ins)), p=p_ws)
            tmp_vertex = ins[tmp_idx]
            tmp_weight = ws[tmp_idx]
        elif mode == 0:
            ins = [edge[indexes.index(direction)] for edge in ins]
            tmp_vertex = np.random.choice(ins)
        if debug:
            print(tmp_vertex)
        output.append(tmp_vertex)
        if return_full_step:
            return output
        else:
            return output[-1]
    else:
        return -1


def diffFunction(prev_edge, new_edges, pressure):
    """
    Function for calculation transition probabilities based on Differences between previous edge and candidate edge
    :param prev_edge: Monetary amount on previous edge
    :param new_edges: Monetary amount on all edges candidates
    :param pressure: The regularization term, higher pressure leads to more strict function
    :return: array of transition probabilities
    """
    return softmax((1.0 - abs(new_edges - prev_edge)) * pressure)


def step(G, vertex, direction="IN", mode=2, allow_back=False, return_full_step=False, pressure=20, debug=False):
    """
     Meta-Random step with changing direction.
    :param G: graph/network on which step should be done
    :param vertex: current vertex
    :param direction: the initial direction of step: IN or OUT
    :param mode: use the edge's weight for transition probability or difference between weights
    :param allow_back: If True, one can get the sequence of the same BPs... Might be delete it?
    :param return_full_step: if True, then step includes intermediate node of FA type
    :param pressure: The regularization term, higher pressure leads to more strict function
    :param debug: print intermediate stages
    :return: next step if succeeded or -1 if failed
    """
    # ////// THE FIRST STEP TO OPPOSITE SET OF NODES \\\\\
    if vertex in [-1, -2, -3]:
        #         Step cannot be made, return -1
        return vertex
    elif not G.has_node(vertex):
        raise ValueError("Vertex {!r} is not in FSN!".format(vertex))
    if direction == "IN":
        ins = G.in_edges(vertex, data=True)
    elif direction == "OUT":
        ins = G.out_edges(vertex, data=True)
    else:
        raise ValueError("Wrong direction argument! {!s} used while IN or OUT are allowed!".format(direction))
    output = list()
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
        output.append(tmp_vertex)
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
            ws = diffFunction(tmp_weight, ws, pressure)
            if debug:
                print(list(zip(outs, ws)))
            tmp_vertex = np.random.choice(outs, p=ws)
            output.append(tmp_vertex)
        elif mode == 1:
            # Transition probability depends on the monetary flows - "rich gets richer"
            ws = ws / np.sum(ws)
            if debug:
                print(list(zip(outs, ws)))
            tmp_vertex = np.random.choice(outs, p=ws)
            output.append(tmp_vertex)
        elif mode == 0:
            # Transition probability is uniform
            if debug:
                print(outs)
            tmp_vertex = np.random.choice(outs)
            output.append(tmp_vertex)
        #     Return next vertex here
        if return_full_step:
            return output
        else:
            return output[-1]
    else:
        return -2


def randomWalk(G, vertex=None, lenght=3, direction="IN", version="MetaDiff", return_full_path=False, debug=False):
    if version not in STEPS_VERSIONS:
        raise ValueError(
            "Given not supported step version {!s}!".format(version) + "\nAllowed only " + str(STEPS_VERSIONS))
    attempts = 10
    context = list()
    if vertex is None:
        context.append(random.choice(list(G.nodes)))
    else:
        context.append(vertex)
    cur_v = context[-1]
    while len(context) < lenght + 1 and attempts > 0:
        try:
            if version == "DefUniform":
                new_v = default_step(G, cur_v, direction, mode=0, return_full_step=return_full_path, debug=debug)
            elif version == "DefWeighted":
                new_v = default_step(G, cur_v, direction, mode=1, return_full_step=return_full_path, debug=debug)
            elif version == "MetaUniform":
                new_v = step(G, cur_v, direction, mode=0, return_full_step=return_full_path, debug=debug)
            elif version == "MetaWeighted":
                new_v = step(G, cur_v, direction, mode=1, return_full_step=return_full_path, debug=debug)
            elif version == "MetaDiff":
                new_v = step(G, cur_v, direction, mode=2, return_full_step=return_full_path, debug=debug)
            attempts -= 1
        except nx.NetworkXError:
            break
        if new_v == -1:
            if debug: print("Cannot continue walking... Termination.")
            break
        if return_full_path:
            context.extend(new_v)
        else:
            context.append(new_v)
        cur_v = context[-1]
    return context
