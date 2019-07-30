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
from NetEmbs import CONFIG
from collections import Counter
import pandas as pd
from NetEmbs.FSN.graph import FSN
import logging
from NetEmbs.CONFIG import LOG, DOUBLE_NEAREST, all_sampling_strategies, PRINT_STATUS, N_JOBS
import time
from pathos.multiprocessing import ProcessPool
import itertools
import os
from NetEmbs.utils.get_size import get_size
from tqdm.auto import tqdm
import pickle

from NetEmbs.FSN import *
np.seterr(all="raise")


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
        ins = list(zip([edge[0] for edge in ins], [edge[-1]["weight"] for edge in ins]))
    elif direction == "OUT":
        ins = G.out_edges(vertex, data=True)
        list(zip([edge[1] for edge in ins], [edge[-1]["weight"] for edge in ins]))
    elif direction == "RANDOM":
        ins = (G.in_edges(vertex, data=True), G.out_edges(vertex, data=True))
        ins = list(zip([edge[0] for edge in ins[0]], [edge[-1]["weight"] for edge in ins[0]])) + \
              list(zip([edge[1] for edge in ins[1]], [edge[-1]["weight"] for edge in ins[1]]))
    else:
        raise ValueError("Wrong direction argument! {!s} used, but IN, OUT or RANDOM are allowed!".format(direction))
    output = list()
    indexes = ["IN", "OUT"]
    # Check that we can make step, otherwise return special value -1
    if len(ins) > 0:
        try:
            # Apply weighted probabilities
            if mode == 1:
                ws = [edge[-1] for edge in ins]
                p_ws = ws / np.sum(ws)
                ins = [edge[0] for edge in ins]
                tmp_idx = np.random.choice(range(len(ins)), p=p_ws)
                tmp_vertex = ins[tmp_idx]
                tmp_weight = ws[tmp_idx]
            #     Apply uniform probabilities
            elif mode == 0:
                ins = [edge[0] for edge in ins]
                tmp_vertex = np.random.choice(ins)
            if debug:
                print(tmp_vertex)
            output.append(tmp_vertex)
        except Exception as e:
            if LOG:
                snapshot = {"CurrentBPNode": vertex,
                            "NextCandidateFA": list(zip(ins, ws))}
                local_logger = logging.getLogger("NetEmbs.Utils.step")
                local_logger.error("Fatal ValueError during 1st sub-step", exc_info=True)
                local_logger.info("Snapshot" + str(snapshot))
    else:
        return -1
    # ///////////// \\\\\\\\\\\\\\\
    # Get the neighborhood of current node regard the chosen direction
    if direction == "IN":
        ins = G.in_edges(tmp_vertex, data=True)
        ins = list(zip([edge[0] for edge in ins], [edge[-1]["weight"] for edge in ins]))
    elif direction == "OUT":
        ins = G.out_edges(tmp_vertex, data=True)
        list(zip([edge[1] for edge in ins], [edge[-1]["weight"] for edge in ins]))
    elif direction == "RANDOM":
        ins = (G.in_edges(tmp_vertex, data=True), G.out_edges(tmp_vertex, data=True))
        ins = list(zip([edge[0] for edge in ins[0]], [edge[-1]["weight"] for edge in ins[0]])) + \
              list(zip([edge[1] for edge in ins[1]], [edge[-1]["weight"] for edge in ins[1]]))
    else:
        raise ValueError("Wrong direction argument! {!s} used, but IN, OUT or RANDOM are allowed!".format(direction))
    # Check that we can make step, otherwise return special value -1
    if len(ins) > 0:
        try:
            if mode == 1:
                ws = [edge[-1] for edge in ins]
                p_ws = ws / np.sum(ws)
                ins = [edge[0] for edge in ins]
                tmp_idx = np.random.choice(range(len(ins)), p=p_ws)
                tmp_vertex = ins[tmp_idx]
                tmp_weight = ws[tmp_idx]
            elif mode == 0:
                ins = [edge[0] for edge in ins]
                tmp_vertex = np.random.choice(ins)
            if debug:
                print(tmp_vertex)
            output.append(tmp_vertex)
        except Exception as e:
            if LOG:
                snapshot = {"CurrentBPNode": vertex,
                            "NextCandidateFA": list(zip(ins, ws))}
                local_logger = logging.getLogger("NetEmbs.Utils.step")
                local_logger.error("Fatal ValueError during 1st sub-step", exc_info=True)
                local_logger.info("Snapshot" + str(snapshot))
        if return_full_step:
            return output
        else:
            return output[-1]
    else:
        return -1


def default_step_old(G, vertex, direction="IN", mode=0, return_full_step=False, debug=False):
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
        raise ValueError("Wrong direction argument! {!s} used, but IN or OUT are allowed!".format(direction))
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


# TODO fix allow_back argument
def step(G, vertex, direction="IN", mode=2, allow_back=False, return_full_step=False, pressure=30,
         debug=False):
    """
     Meta-Random step with changing direction.
    :param G: graph/network on which step should be done
    :param vertex: current vertex
    :param direction: the initial direction of step: IN or OUT
    :param mode: use the edge's weight for transition probability or difference between weights
    :param allow_back: If True, one can get the sequence of the same BPs... Might be delete it? TODO check, is it needed?
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
        try:
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
        except Exception as e:
            if LOG:
                snapshot = {"CurrentBPNode": vertex,
                            "NextCandidateFA": list(zip(ins, ws))}
                local_logger = logging.getLogger("NetEmbs.Utils.step")
                local_logger.error("Fatal ValueError during 1st sub-step", exc_info=True)
                local_logger.info("Snapshot" + str(snapshot))
                local_logger = logging.getLogger(CONFIG.MAIN_LOGGER + ".Utils.step")
                local_logger.error("Fatal ValueError during 1st sub-step", exc_info=True)
                local_logger.info("Snapshot" + str(snapshot))
        #     Return next vertex here
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
        probas = None
        try:
            if mode == 2:
                # Transition probability depends on the difference between monetary flows
                probas = diff_function(tmp_weight, ws, pressure)
                if debug:
                    print(f"Pressure is {pressure}, weight is {tmp_weight}")
                    print(list(zip(outs, list(zip(ws, probas)))))
                tmp_vertex = np.random.choice(outs, p=probas)
                output.append(tmp_vertex)
            elif mode == 1:
                # Transition probability depends on the monetary flows - "rich gets richer"
                probas = ws / np.sum(ws)
                if debug:
                    print(list(zip(outs, ws)))
                tmp_vertex = np.random.choice(outs, p=probas)
                output.append(tmp_vertex)
            elif mode == 0:
                # Transition probability is uniform
                if debug:
                    print(outs)
                tmp_vertex = np.random.choice(outs)
                output.append(tmp_vertex)
        except Exception as e:
            if LOG:
                snapshot = {"CurrentNode": tmp_vertex, "CurrentWeight": tmp_weight,
                            "NextCandidates": list(zip(outs, ws)), "Probas": probas}
                # Local logger
                local_logger = logging.getLogger("NetEmbs.Utils.step")
                local_logger.error("Fatal ValueError during 2nd sub-step", exc_info=True)
                local_logger.info("Snapshot" + str(snapshot))
                #         Global logger
                local_logger = logging.getLogger(CONFIG.MAIN_LOGGER + ".Utils.step")
                local_logger.error("Fatal ValueError during 2nd sub-step", exc_info=True)
                local_logger.info("Snapshot" + str(snapshot))
        #     Return next vertex here
        if return_full_step:
            return output
        else:
            return output[-1]
    else:
        return -2


def randomWalk(G, vertex=None, length=10, direction="IN", pressure=30, version="MetaDiff", return_full_path=False,
               debug=False):
    """
    RandomWalk th
    :param G: Bipartite graph, an instance of networkx
    :param vertex: initial node
    :param length: the maximum length of RandomWalk
    :param direction: The direction of walking. IN - go via source financial accounts, OUT - go via target financial accounts
    :param pressure: The strength of selection process during walking
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
    if version not in all_sampling_strategies:
        raise ValueError(
            "Given not supported step version {!s}!".format(version) + "\nAllowed only " + str(all_sampling_strategies))
    context = list()
    if vertex is None:
        context.append(random.choice(list(G.nodes)))
    else:
        context.append(vertex)
    cur_v = context[-1]
    mask = {"IN": "OUT", "OUT": "IN"}
    cur_direction = "IN"
    while len(context) < length:
        try:
            # TODO, June 2, here
            if version == "DefUniform":
                if direction == "COMBI":
                    new_v = default_step(G, cur_v, cur_direction, mode=0, return_full_step=return_full_path,
                                         debug=debug)
                    cur_direction = mask[cur_direction]
                else:
                    new_v = default_step(G, cur_v, direction, mode=0, return_full_step=return_full_path, debug=debug)
                if new_v == -1:
                    # No edges in the set direction...
                    if debug: print("Cannot continue walking... Termination.")
                    break
            elif version == "DefWeighted":
                if direction == "COMBI":
                    new_v = default_step(G, cur_v, cur_direction, mode=1, return_full_step=return_full_path,
                                         debug=debug)
                    cur_direction = mask[cur_direction]
                else:
                    new_v = default_step(G, cur_v, direction, mode=1, return_full_step=return_full_path, debug=debug)
                if new_v == -1:
                    # No edges in the set direction...
                    if debug: print("Cannot continue walking... Termination.")
                    break
            elif version == "MetaUniform":
                if direction == "COMBI":
                    new_v = step(G, cur_v, cur_direction, pressure=pressure, mode=0, return_full_step=return_full_path,
                                 debug=debug)
                    cur_direction = mask[cur_direction]
                else:
                    new_v = step(G, cur_v, direction, pressure=pressure, mode=0, return_full_step=return_full_path,
                                 debug=debug)
            elif version == "MetaWeighted":
                new_v = step(G, cur_v, direction, pressure=pressure, mode=1, return_full_step=return_full_path,
                             debug=debug)
            elif version == "MetaDiff":
                if direction == "TRIPLE":
                    pass
                elif direction == "COMBI":
                    new_v = new_step(G, cur_v, cur_direction, pressure=pressure)
                    # new_v = step(G, cur_v, cur_direction, pressure=pressure, mode=2, return_full_step=return_full_path,
                    #              debug=debug)
                    cur_direction = mask[cur_direction]
                else:
                    new_v = step(G, cur_v, direction, pressure=pressure, mode=2, return_full_step=return_full_path,
                                 debug=debug)
            elif version == "OriginalRandomWalk":
                #         The direct implementation of Perozi randomWalk
                new_v = default_step(G, cur_v, direction="RANDOM", mode=0, return_full_step=return_full_path,
                                     debug=debug)

        except nx.NetworkXError:
            # TODO modify to more robust behaviour
            break
        if new_v == -1:
            # No edges in the set direction...
            if direction != "COMBI":
                if debug: print("Cannot continue walking... Termination.")
                break
            else:
                if debug: print("Cannot continue walking... Change direction.")
                cur_v = context[-1]
        else:
            if return_full_path:
                if isinstance(new_v, list):
                    context.extend(new_v)
                else:
                    context.append(new_v)
            else:
                context.append(new_v)
            #     TODO modification is here! check is it needed or not
            if DOUBLE_NEAREST:
                context.extend(context[-2:])
            cur_v = context[-1]
    return context


# Feed the CONFIG values into each sampling methods within Pathos multiprocessing
def wrappedRandomWalk(node):
    return [randomWalk(CONFIG.GLOBAL_FSN, node, length=CONFIG.WALKS_LENGTH, pressure=CONFIG.PRESSURE,
                       direction=CONFIG.DIRECTION,
                       version=CONFIG.STRATEGY)
            for _
            in range(CONFIG.WALKS_PER_NODE)]


def wrappedRandomWalkIN(node):
    return [randomWalk(CONFIG.GLOBAL_FSN, node, length=CONFIG.WALKS_LENGTH, pressure=CONFIG.PRESSURE,
                       direction="IN", version=CONFIG.STRATEGY)
            for _
            in range(CONFIG.WALKS_PER_NODE)]


def wrappedRandomWalkOUT(node):
    return [
        randomWalk(CONFIG.GLOBAL_FSN, node, length=CONFIG.WALKS_LENGTH, pressure=CONFIG.PRESSURE,
                   direction="OUT", version=CONFIG.STRATEGY)
        for _
        in range(CONFIG.WALKS_PER_NODE)]


def wrappedOriginalRandomWalk(node):
    return [
        randomWalk(CONFIG.GLOBAL_FSN, node, length=CONFIG.WALKS_LENGTH, pressure=CONFIG.PRESSURE,
                   direction="RANDOM", version=CONFIG.STRATEGY)
        for _
        in range(CONFIG.WALKS_PER_NODE)]


def graph_sampling(n_jobs=4, direction=None):
    """
    Construction a sequences of nodes from given FSN
    :param n_jobs: Number of parallel processes to be created
    :param direction: initial direction
    :return: array of sampled nodes
    """
    if direction is None:
        direction = CONFIG.DIRECTION
    max_processes = max(n_jobs, os.cpu_count())
    pool = ProcessPool(nodes=max_processes)
    # required to restart pool to update CONFIG inside the parallel part
    pool.terminate()
    pool.restart()
    BPs = CONFIG.GLOBAL_FSN.get_BPs()
    n_BPs = len(BPs)
    sampled = list()
    if LOG:
        local_logger = logging.getLogger("NetEmbs.Utils.graph_sampling")
        local_logger.info("Created a Pool with " + str(max_processes) + " processes ")
        local_logger.info("Total size of broadcasting arguments is " + str(get_size(BPs)) + " bytes ")
        local_logger.info("Total size of FSN is " + str(get_size(CONFIG.GLOBAL_FSN)) + " bytes ")

    if direction not in ["ALL", "IN", "OUT", "COMBI", "RANDOM"]:
        raise ValueError(
            "Given not supported yet direction of walking {!s}!".format(direction) + "\nAllowed only " + str(
                ["ALL", "IN", "OUT"]))
    if direction == "ALL":
        print("Chosen ALL direction, hence, run both IN and OUT randomWalks from each node...")
        if LOG:
            local_logger = logging.getLogger("NetEmbs.Utils.graph_sampling")
            local_logger.info("Chosen direction ALL, hence, run both IN and OUT randomWalks from each node! ")
        try:
            with tqdm(total=n_BPs) as pbar:
                for i, res in enumerate(pool.uimap(wrappedRandomWalkIN, BPs)):
                    sampled.append(res)
                    pbar.update()
        except KeyboardInterrupt:
            print('got ^C while pool mapping, terminating the pool')
            pool.terminate()
        pool.terminate()
        pool.restart()
        print("Done with IN direction!")
        try:
            with tqdm(total=n_BPs) as pbar:
                for i, res in enumerate(pool.uimap(wrappedRandomWalkOUT, BPs)):
                    sampled.append(res)
                    pbar.update()
        except KeyboardInterrupt:
            print('got ^C while pool mapping, terminating the pool')
            pool.terminate()
    elif direction in ["COMBI", "IN", "OUT", "RANDOM"]:
        # sampled = [wrappedRandomWalk(node) for node in tqdm(GLOBAL_FSN.get_BP())]
        try:
            with tqdm(total=n_BPs) as pbar:
                for i, res in enumerate(pool.uimap(wrappedRandomWalk, BPs)):
                    sampled.append(res)
                    pbar.update()
        except KeyboardInterrupt:
            print('got ^C while pool mapping, terminating the pool')
            pool.terminate()
    res = list(itertools.chain(*sampled))
    pool.terminate()
    pool.restart()
    if LOG:
        local_logger = logging.getLogger("NetEmbs.Utils.graph_sampling")
        local_logger.info("Total number of raw sampled sequences is " + str(len(res)))
        local_logger.info("Average length of sequences is " + str(sum(map(len, res)) / float(len(res))))
    return res


def make_pairs(sampled_seq):
    """
    Helper function for construction pairs from sequence of nodes with given window size
    :param sampled_seq: Original sequence of nodes (output of RandomWalk procedure)
    !NOTE: Window size is used from CONFIG file for better performance!
    :return:
    """
    output = list()
    try:
        for cur_idx in range(len(sampled_seq)):
            for drift in range(max(0, cur_idx - CONFIG.WINDOW_SIZE),
                               min(cur_idx + CONFIG.WINDOW_SIZE + 1, len(sampled_seq))):
                if drift != cur_idx:
                    output.append((sampled_seq[cur_idx], sampled_seq[drift]))
        return output
    except TypeError:
        print("t")


def get_pairs(n_jobs=4, direction=CONFIG.DIRECTION, drop_duplicates=True, use_cache=True):
    """
    Construction a pairs (skip-grams) of nodes according to sampled sequences
    :param n_jobs: Number of parallel processes to be created
    :param direction: initial direction
    :param drop_duplicates: True, delete pairs with equal elements
    :param use_cache: True, then try to find cached Skip-Grams. Allows to save ~80% of execution time.
    :return: array of pairs(joint appearance of two BP nodes)
    """
    if direction not in ["ALL", "IN", "OUT", "COMBI", "RANDOM"]:
        raise ValueError(
            "Given not supported yet direction of walking {!s}!".format(direction) + "\nAllowed only " + str(
                ["ALL", "IN", "OUT", "COMBI", "RANDOM"]))
    if not use_cache:
        if PRINT_STATUS:
            print("--------- Started the SAMPLING the sequences from FSN ---------")

        start_time = time.time()
        sequences = graph_sampling(n_jobs)
        if CONFIG.HACK:
            #             Explicitly sample the 1-hop neighbours
            _tmps = (CONFIG.WALKS_LENGTH, CONFIG.WALKS_PER_NODE)
            CONFIG.WALKS_PER_NODE = int(
                CONFIG.HACK * CONFIG.WALKS_PER_NODE * CONFIG.WALKS_LENGTH * CONFIG.WALKS_LENGTH / CONFIG.WINDOW_SIZE)
            CONFIG.WALKS_LENGTH = 2
            print("Additionally sample the nearest neighbours...")
            sequences.extend(graph_sampling(n_jobs))
            CONFIG.WALKS_LENGTH = _tmps[0]
            CONFIG.WALKS_PER_NODE = _tmps[1]
        end_time = time.time()
        print("Elapsed time for sampling: ", end_time - start_time)
        print("Cashing sampled sequences...")
        with open(CONFIG.WORK_FOLDER[0] + "sampled_sequences_cached.pkl", "wb") as file:
            pickle.dump(sequences, file)
    elif use_cache:
        print("Loading sequences from cache... wait...")
        try:
            with open(CONFIG.WORK_FOLDER[0] + "sampled_sequences_cached.pkl", "rb") as file:
                sequences = pickle.load(file)
        except FileNotFoundError:
            print("File not found... Recalculate \n")
            print("Sampling sequences... wait...")
            start_time = time.time()
            sequences = graph_sampling(n_jobs)
            if CONFIG.HACK:
                #             Explicitly sample the 1-hop neighbours
                _tmps = (CONFIG.WALKS_LENGTH, CONFIG.WALKS_PER_NODE)
                CONFIG.WALKS_PER_NODE = int(
                    CONFIG.HACK * CONFIG.WALKS_PER_NODE * CONFIG.WALKS_LENGTH * CONFIG.WALKS_LENGTH / CONFIG.WINDOW_SIZE)
                CONFIG.WALKS_LENGTH = 2
                print("Additionally sample the nearest neighbours...")
                sequences.extend(graph_sampling(n_jobs))
                CONFIG.WALKS_LENGTH = _tmps[0]
                CONFIG.WALKS_PER_NODE = _tmps[1]
            end_time = time.time()
            print("Elapsed time for sampling: ", end_time - start_time)
            print("Cashing sampled sequences...")
            with open(CONFIG.WORK_FOLDER[0] + "sampled_sequences_cached.pkl", "wb") as file:
                pickle.dump(sequences, file)
    if PRINT_STATUS:
        print("--------- Ended the SAMPLING the sequences from FSN ---------")
    max_processes = max(n_jobs, os.cpu_count())
    pool_pairs = ProcessPool(nodes=max_processes)
    pool_pairs.terminate()
    pool_pairs.restart()
    if PRINT_STATUS:
        print("--------- Started making pairs from the sequences ---------")
    pairs = pool_pairs.map(make_pairs, sequences)
    logging.getLogger("NetEmbs.utils.get_pairs").info("Total number of raw sampled pairs is " + str(len(pairs)))
    if PRINT_STATUS:
        print("--------- Ended making pairs from the sequences ---------")
    if drop_duplicates:
        pairs = [item for sublist in pairs for item in sublist if item[0] != item[1]]
    else:
        pairs = [item for sublist in pairs for item in sublist]
    pairs = [item for item in pairs if (item[0]!=-3) & (item[1]!=-3)]
    pool_pairs.terminate()
    pool_pairs.restart()
    return pairs


def get_top_similar(all_pairs, top=3, as_DataFrame=True, sort_ids=True, title="Similar_BP"):
    """
    Helper function for counting joint appearance of nodes and returning top N
    :param all_pairs: all found pairs
    :param top: required number of top elements for each node
    :param as_DataFrame: convert output to DataFrame
    :param sort_ids: Sort output DataFrame w.r.t. ID column
    :param title: title of column in returned DataFrame
    :return: dictionary with node number as a key and values as list[node, cnt]
    """
    per_node = {item[0]: list() for item in all_pairs}
    output_top = dict()
    for item in all_pairs:
        per_node[item[0]].append(item[1])
    for key, data in per_node.items():
        output_top[key] = Counter(per_node[key]).most_common(top)
    if as_DataFrame:
        if sort_ids:
            return pd.DataFrame(output_top.items(), columns=["ID", title]).sort_values(by=["ID"])
        else:
            return pd.DataFrame(output_top.items(), columns=["ID", title])
    else:
        return output_top


def get_SkipGrams(df=None, version=None, walk_length=None, walks_per_node=None, direction=None, use_cache=False):
    """
    Get Skip-Grams for given DataFrame with Entries records
    :param df: original DataFrame
    :param version: Version of step:
    "DefUniform" - Pure RandomWalk (uniform probabilities, follows the direction),
    "DefWeighted" - RandomWalk (weighted probabilities, follows the direction),
    "MetaUniform" - Default Metapath-version (uniform probabilities, change directions),
    "MetaWeighted" - Weighted Metapath version (weighted probabilities "rich gets richer", change directions),
    "MetaDiff" - Modified Metapath version (probabilities depend on the differences between edges, change directions)
    :param walk_length: max length of RandomWalk
    :param walks_per_node: max number of RandomWalks per each node in FSN
    :param direction: initial direction
    :param use_cache: If True, cache the intermediate SkipGrams sequences
    :return: list of all pairs
    :return fsn: FSN class instance for given DataFrame
    :return tr: Encoder/Decoder for given DataFrame
    """

    if CONFIG.GLOBAL_FSN is None and isinstance(df, pd.DataFrame):
        #     Cannot find already existing FSN object, construct new one
        print("No FSN object in memory. Construct from scratch!")
        try:
            CONFIG.GLOBAL_FSN = FSN()
            CONFIG.GLOBAL_FSN.build(df, left_title="FA_Name")
        except ValueError:
            raise RuntimeError(
                f"Could not interpret the given argument: df is type of {type(df)}, while pd.DataFrame required! ")
    elif isinstance(df, FSN):
        CONFIG.GLOBAL_FSN = df
    elif isinstance(CONFIG.GLOBAL_FSN, FSN):
        pass
    else:
        raise TypeError("Cound not find FSN in memory as well as construct from scratch!")
    tr = TransformationBPs(CONFIG.GLOBAL_FSN.get_BPs())

    # //////// UPDATE CONFIG IF NEEDED w.r.t the given arguments \\\\\\\\\\\
    if walks_per_node is not None:
        CONFIG.WALKS_PER_NODE = walks_per_node
    if walk_length is not None:
        CONFIG.WALKS_LENGTH = walk_length
    if direction is not None:
        CONFIG.DIRECTION = direction
    if version is not None:
        CONFIG.STRATEGY = version

    print(f"Current SAMPLING parameters: \n WindowSize:  {CONFIG.WINDOW_SIZE} \n Pressure:  {CONFIG.PRESSURE}"
          f"\n Strategy:  {CONFIG.STRATEGY}")
    logging.getLogger(CONFIG.MAIN_LOGGER + ".FSN.utils").info(
        f"Current SAMPLING parameters: \n WindowSize:  {CONFIG.WINDOW_SIZE} \n Pressure:  {CONFIG.PRESSURE}"
        f"\n Strategy:  {CONFIG.STRATEGY}")
    # TODO before that stage, all CONFIG has to be updated!
    if not use_cache:
        print("Start sampling... wait...")
        skip_gr = tr.encode_pairs(
            get_pairs(N_JOBS, CONFIG.DIRECTION, use_cache=use_cache))
        with open(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + "skip_grams_cached.pkl", "wb") as file:
            pickle.dump(skip_gr, file)
        print("Sampled SkipGrams are saved in cache... Total size is ", get_size(skip_gr), " bytes")
    elif use_cache:
        print("Loading SkipGrams from cache... wait...")
        try:
            with open(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + "skip_grams_cached.pkl", "rb") as file:
                skip_gr = pickle.load(file)
        except FileNotFoundError:
            print("File not found... Recalculate \n")
            print("Start sampling... wait...")
            skip_gr = tr.encode_pairs(
                get_pairs(N_JOBS, CONFIG.DIRECTION, use_cache=use_cache))
            with open(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + "skip_grams_cached.pkl", "wb") as file:
                pickle.dump(skip_gr, file)
    else:
        raise ValueError(
            "Use True or False for skip_gr argument! {!s}!".format(use_cache) + " was given")
    logging.getLogger("NetEmbs.utils.get_SkipGrams").info("Total number of GOOD sampled pairs is " + str(len(skip_gr)))
    return skip_gr, CONFIG.GLOBAL_FSN, tr


def get_SkipGrams_raw(df, version="MetaDiff", walk_length=10, walks_per_node=10, direction="COMBI", use_cache=False):
    """
    Get Skip-Grams for given DataFrame with Entries records
    :param df: original DataFrame
    :param version: Version of step:
    "DefUniform" - Pure RandomWalk (uniform probabilities, follows the direction),
    "DefWeighted" - RandomWalk (weighted probabilities, follows the direction),
    "MetaUniform" - Default Metapath-version (uniform probabilities, change directions),
    "MetaWeighted" - Weighted Metapath version (weighted probabilities "rich gets richer", change directions),
    "MetaDiff" - Modified Metapath version (probabilities depend on the differences between edges, change directions)
    :param walk_length: max length of RandomWalk
    :param walks_per_node: max number of RandomWalks per each node in FSN
    :param direction: initial direction
    :param use_cache: If True, cache the intermediate SkipGrams sequences
    :return: list of all pairs
    :return fsn: FSN class instance for given DataFrame
    :return tr: Encoder/Decoder for given DataFrame
    """
    # TODO check current version vs. CONFIG.GLOBAL_FSN

    CONFIG.GLOBAL_FSN = FSN()
    CONFIG.GLOBAL_FSN.build(df, left_title="FA_Name")
    # //////// UPDATE CONFIG IF NEEDED w.r.t the given arguments \\\\\\\\\\\
    if walks_per_node is not None:
        CONFIG.WALKS_PER_NODE = walks_per_node
    if walk_length is not None:
        CONFIG.WALKS_LENGTH = walk_length
    if direction is not None:
        CONFIG.DIRECTION = direction
    if version is not None:
        CONFIG.STRATEGY = version

    if not use_cache:
        print("Start sampling... wait...")
        skip_gr = get_pairs(N_JOBS, CONFIG.DIRECTION, use_cache=use_cache)
        with open(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + "skip_grams_cached.pkl", "wb") as file:
            pickle.dump(skip_gr, file)
        print("Sampled SkipGrams are saved in cache... Total size is ", get_size(skip_gr), " bytes")
    elif use_cache:
        print("Loading SkipGrams from cache... wait...")
        try:
            with open(CONFIG.WORK_FOLDER[0] + "skip_grams_cached.pkl", "rb") as file:
                skip_gr = pickle.load(file)
        except FileNotFoundError:
            print("File not found... Recalculate \n")
            print("Start sampling... wait...")
            skip_gr = get_pairs(N_JOBS, CONFIG.DIRECTION, use_cache=use_cache)
            with open(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + "skip_grams_cached.pkl", "wb") as file:
                pickle.dump(skip_gr, file)
    else:
        raise ValueError(
            "Use True or False for skip_gr argument! {!s}!".format(use_cache) + " was given")
    return skip_gr


class TransformationBPs:
    """
    Encode/Decode original BP nodes number to/from sequential integers for TensorFlow
    """

    def __init__(self, original_bps):
        self.len = len(original_bps)
        self.original_bps = original_bps
        self._enc_dec()

    def _enc_dec(self):
        self.encoder = dict(list(zip(self.original_bps, range(self.len))))
        self.decoder = dict(list(zip(range(self.len), self.original_bps)))

    def encode(self, original_seq):
        return [self.encoder[item] for item in original_seq]

    def decode(self, seq):
        return [self.decoder[item] for item in seq]

    def encode_pairs(self, original_pairs):
        return [(self.encoder[item[0]], self.encoder[item[1]]) for item in original_pairs]

    def decode_pairs(self, encoded_pairs):
        return [(self.decoder[item[0]], self.decoder[item[1]]) for item in encoded_pairs]


def find_similar(df, top_n=3, version="MetaDiff", walk_length=10, walks_per_node=10, direction="IN",
                 column_title="Similar_BP"):
    fsn = FSN()
    fsn.build(df, left_title="FA_Name")
    if LOG:
        local_logger = logging.getLogger("NetEmbs.Utils.find_similar")
    if not isinstance(version, list) and not isinstance(direction, list):
        pairs = get_pairs(fsn, N_JOBS)
        return get_top_similar(pairs, top=top_n, title=column_title)
    else:
        #         Multiple parameters, build grid over them
        if not isinstance(version, list) and isinstance(version, str):
            version = [version]
        if not isinstance(direction, list) and isinstance(direction, str):
            direction = [direction]
        #             All possible combinations:
        _first = True
        for ver in version:
            for _dir in direction:
                if LOG:
                    local_logger.info("Current arguments are " + ver + " and " + _dir)
                if _first:
                    _first = False
                    output_df = get_top_similar(
                        get_pairs(fsn, N_JOBS), top=top_n, title=str(ver + "_" + _dir))
                else:
                    output_df[str(ver + "_" + _dir)] = get_top_similar(
                        get_pairs(fsn, N_JOBS), top=top_n, title=str(ver + "_" + _dir))[str(ver + "_" + _dir)]
        return output_df


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


def get_JournalEntries(df):
    """
    Helper function for extraction Journal Entries from Entry Records DataFrame
    :param df: Original DataFrame with Entries Records
    :return: Journal Entries DataFrame, each row is separate business process
    """
    if "Signature" not in list(df):
        from NetEmbs.DataProcessing.unique_signatures import leave_unique_business_processes
        df = leave_unique_business_processes(df)
    return df[["ID", "Signature"]].drop_duplicates("ID")


global journal_decoder


def decode_row(row):
    global journal_decoder
    output = dict()
    output["ID"] = row["ID"]
    output["Signature"] = row["Signature"]
    for cur_title in row.index._data[2:]:
        cur_row_decoded = list()
        if row[cur_title] == -1.0:
            output[cur_title] = None
        else:
            for item in row[cur_title]:
                cur_row_decoded.append(journal_decoder[item[0]])
                cur_row_decoded.append("---------")
            output[cur_title] = cur_row_decoded

    return pd.Series(output)


def similar(df, top_n=3, version="MetaDiff", walk_length=10, walks_per_node=10, direction=["IN", "OUT", "COMBI"]):
    """
    Finding "similar" BP
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
    global journal_decoder
    if LOG:
        local_logger = logging.getLogger("NetEmbs.Utils.Similar")
        local_logger.info("Given directions are " + str(direction))
        local_logger.info("Given versions are " + str(version))
    journal_entries = get_JournalEntries(df)

    if LOG:
        local_logger.info("Journal entries have been extracted!")
    journal_decoder = journal_entries.set_index("ID").to_dict()["Signature"]
    print("Done with extraction Journal Entries data!")
    output = find_similar(df, top_n=top_n, version=version, walk_length=walk_length, walks_per_node=walks_per_node,
                          direction=direction)
    print("Done with RandomWalking... Found ", str(top_n), " top")
    journal_entries = journal_entries.merge(output,
                                            on="ID", how="left")
    journal_entries.fillna(-1.0, inplace=True)
    res = journal_entries.apply(decode_row, axis=1)
    return res
