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
from collections import Counter
import pandas as pd
from NetEmbs.FSN.graph import FSN


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


def diff_function(prev_edge, new_edges, pressure):
    """
    Function for calculation transition probabilities based on Differences between previous edge and candidate edge
    :param prev_edge: Monetary amount on previous edge
    :param new_edges: Monetary amount on all edges candidates
    :param pressure: The regularization term, higher pressure leads to more strict function
    :return: array of transition probabilities
    """
    return softmax((1.0 - abs(new_edges - prev_edge)) * pressure)


def make_pairs(sampled_seq, window=3, debug=False):
    """
    Helper function for construction pairs from sequence of nodes with given window size
    :param sampled_seq: Original sequence of nodes (output of RandomWalk procedure)
    :param window: window size, how much predecessors and successors one takes into account
    :param debug: print intermediate stages
    :return:
    """
    if debug:
        print(sampled_seq)
    output = list()
    for cur_idx in range(len(sampled_seq)):
        for drift in range(max(0, cur_idx - window), min(cur_idx + window + 1, len(sampled_seq))):
            if drift != cur_idx:
                output.append((sampled_seq[cur_idx], sampled_seq[drift]))
    if len(output) < 2:
        print(output)
    return output


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
            ws = diff_function(tmp_weight, ws, pressure)
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


def randomWalk(G, vertex=None, length=3, direction="IN", version="MetaDiff", return_full_path=False, debug=False):
    """
    RandomWalk function for sampling the sequence of nodes from given graph and initial node
    :param G: Bipartite graph, an instance of networkx
    :param vertex: initial node
    :param length: the maximum length of RandomWalk
    :param direction: The direction of walking. IN - go via source financial accounts, OUT - go via target financial accounts
    :param version: Version of step:
    "DefUniform" - Pure RandomWalk (uniform probabilities, follows the direction),
    "DefWeighted" - RandomWalk (weighted probabilities, follows the direction),
    "MetaUniform" - Default Metapath-version (uniform probabilities, change directions),
    "MetaWeighted" - Weighted Metapath version (weighted probabilities "rich gets richer", change directions),
    "MetaDiff" - Modified Metapath version (probabilities depend on the differences between edges, change directions)
    :param return_full_path: If True, return the full path with FA nodes
    :param debug: Debug boolean flag, print intermediate steps
    :return: Sampled sequence of nodes
    """
    if version not in STEPS_VERSIONS:
        raise ValueError(
            "Given not supported step version {!s}!".format(version) + "\nAllowed only " + str(STEPS_VERSIONS))
    context = list()
    if vertex is None:
        context.append(random.choice(list(G.nodes)))
    else:
        context.append(vertex)
    cur_v = context[-1]
    while len(context) < length + 1:
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
        except nx.NetworkXError:
            # TODO modify to more robust behaviour
            break
        if new_v == -1:
            if debug: print("Cannot continue walking... Termination.")
            break
        if return_full_path:
            if isinstance(new_v, list):
                context.extend(new_v)
            else:
                context.append(new_v)
        else:
            context.append(new_v)
        cur_v = context[-1]
    return context


def get_pairs(fsn, version="MetaDiff", walk_length=10, walks_per_node=10, direction="IN", drop_duplicates=True):
    """
    Construction a pairs (skip-grams) of nodes according to sampled sequences
    :param fsn: Researched FSN
    :param version: Applying version of step method
    "DefUniform" - Pure RandomWalk (uniform probabilities, follows the direction),
    "DefWeighted" - RandomWalk (weighted probabilities, follows the direction),
    "MetaUniform" - Default Metapath-version (uniform probabilities, change directions),
    "MetaWeighted" - Weighted Metapath version (weighted probabilities "rich gets richer", change directions),
    "MetaDiff" - Modified Metapath version (probabilities depend on the differences between edges, change directions)
    :param walk_length: max length of RandomWalk
    :param walks_per_node: max number of RandomWalks per each node in FSN
    :param direction: initial direction
    :param drop_duplicates: True, delete pairs with equal elements
    :return: array of pairs(joint appearance of two BP nodes)
    """
    pairs = [make_pairs(randomWalk(fsn, node, walk_length, direction="IN", version=version)) for _ in
             range(walks_per_node) for node
             in fsn.get_BP()] + [make_pairs(randomWalk(fsn, node, walk_length, direction="OUT", version=version)) for _
                                 in
                                 range(walks_per_node) for node
                                 in fsn.get_BP()]
    if drop_duplicates:
        pairs = [item for sublist in pairs for item in sublist if item[0] != item[1]]
    else:
        pairs = [item for sublist in pairs for item in sublist]
    return pairs


def get_top_similar(all_pairs, top=3, as_DataFrame=True):
    """
    Helper function for counting joint appearance of nodes and returning top N
    :param all_pairs: all found pairs
    :param top: required number of top elements for each node
    :return: dictionary with node number as a key and values as list[node, cnt]
    """
    per_node = {item[0]: list() for item in all_pairs}
    output_top = dict()
    for item in all_pairs:
        per_node[item[0]].append(item[1])
    for key, data in per_node.items():
        output_top[key] = Counter(per_node[key]).most_common(top)
    if as_DataFrame:
        return pd.DataFrame(output_top.items(), columns=["ID", "Similar_BP"])
    else:
        return output_top


def find_similar(df, top_n=3, version="MetaDiff", walk_length=10, walks_per_node=10, direction="IN"):
    fsn = FSN()
    fsn.build(df, name_column="FA_Name")
    pairs = get_pairs(fsn, version=version, walk_length=walk_length, walks_per_node=walks_per_node, direction=direction)
    return get_top_similar(pairs, top=top_n)


def add_similar(df, top_n=3, version="MetaDiff", walk_length=10, walks_per_node=10, direction="IN"):
    """
    Adding "similar" BP
    :param df: original DataFrame
    :param top_n: the number of BP to store
    :param version: Version of step:
    "DefUniform" - Pure RandomWalk (uniform probabilities, follows the direction),
    "DefWeighted" - RandomWalk (weighted probabilities, follows the direction),
    "MetaUniform" - Default Metapath-version (uniform probabilities, change directions),
    "MetaWeighted" - Weighted Metapath version (weighted probabilities "rich gets richer", change directions),
    "MetaDiff" - Modified Metapath version (probabilities depend on the differences between edges, change directions)
    :param walk_length: max length of RandomWalk
    :param walks_per_node: max number of RandomWalks per each node in FSN
    :param direction: initial direction
    :return: original DataFrame with Similar_BP column
    """
    return df.merge(
        find_similar(df, top_n=top_n, version=version, walk_length=walk_length, walks_per_node=walks_per_node,
                     direction=direction),
        on="ID", how="left")
