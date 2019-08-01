# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
sampling.py
Created by lex at 2019-08-01.
"""
from NetEmbs.GraphSampling.walk_strategies import *
import numpy as np
import random
from NetEmbs.FSN.graph import FSN
from NetEmbs.utils.Logs.make_snapshot import log_snapshot
from NetEmbs import CONFIG
from typing import Union, Tuple, Optional, List
import os
import time
from pathos.multiprocessing import ProcessPool
import itertools
import os
import logging
import pickle
from tqdm.auto import tqdm

strategy_to_class = {"MetaDiff": finWalk, "MetaUniform": metaUniform, "MetaWeighted": metaWeighted,
                     "DefUniform": defUniform, "DefWeighted": defWeighted, "OriginalRandomWalk": originalRandomWalk}


def wrappedWalk(node: Union[str, int]) -> List[Union[str, int]]:
    """
    Wrapper around walk methods to be executed in parallel with Pathod
    Parameters
    ----------
    node : str or int
        Initial node from where to start sampling

    Returns
    -------
        The sampled sequences of nodes
    """
    global walk
    return [walk.walk(node) for _ in range(CONFIG.WALKS_PER_NODE)]


def graph_sampling(strategy: Optional[str] = "MetaDiff", n_jobs: Optional[int] = 4,
                   use_cache: Optional[bool] = True) -> List[List[Union[str, int]]]:
    """
    Sampling the sequences of nodes from FSN w.r.t. chosen strategy
    Parameters
    ----------
    strategy : str, default is 'MetaDiff'
        Walking strategy to be used
    n_jobs : int, default is 4
        Number of workers to be created in parallel pool
    use_cache : book, default is True
        To use the previously cached files

    Returns
    -------
    Sampled sequences of BP nodes
    """
    local_logger = logging.getLogger(f"{__name__}")
    if use_cache and os.path.isfile(CONFIG.WORK_FOLDER[0] + "sampled_sequences_cached.pkl"):
        local_logger.info("Loading sequences from cache... wait...")
        try:
            with open(CONFIG.WORK_FOLDER[0] + "sampled_sequences_cached.pkl", "rb") as file:
                res = pickle.load(file)
        except FileNotFoundError:
            local_logger.info("File not found... Recalculate \n")
            pass
    else:
        local_logger.info("Sampling sequences... wait...")
        max_processes = max(n_jobs, os.cpu_count())
        global walk
        if strategy in strategy_to_class.keys():
            walk = strategy_to_class[strategy](G=CONFIG.GLOBAL_FSN, walk_length=CONFIG.WALKS_LENGTH,
                                               direction=CONFIG.DIRECTION,
                                               pressure=CONFIG.PRESSURE, allow_back=CONFIG.ALLOW_BACK)
        else:
            raise KeyError(
                f"The given strategy {strategy} is unknown. The following ones are implemented: {strategy_to_class.keys()}")
        sampling_pool = ProcessPool(nodes=max_processes)
        local_logger.info("Created a Pool with " + str(max_processes) + " processes ")
        # required to restart pool to update CONFIG inside the parallel part
        sampling_pool.terminate()
        sampling_pool.restart()
        BPs = CONFIG.GLOBAL_FSN.get_BPs()
        n_BPs = len(BPs)
        sampled = list()
        try:
            with tqdm(total=n_BPs) as pbar:
                for i, res in enumerate(sampling_pool.uimap(wrappedWalk, BPs)):
                    sampled.append(res)
                    pbar.update()
        except KeyboardInterrupt:
            print('got ^C while pool mapping, terminating the pool')
            sampling_pool.terminate()
        res = list(itertools.chain(*sampled))
        sampling_pool.terminate()
        sampling_pool.restart()
        local_logger.info("Cashing sampled sequences!")
        with open(CONFIG.WORK_FOLDER[0] + "sampled_sequences_cached.pkl", "wb") as file:
            pickle.dump(res, file)
    local_logger.info(f"Total number of raw sampled sequences is {len(res)}")
    local_logger.info(f"Average length of sequences is {sum(map(len, res)) / float(len(res))}")
    return res


def _make_pairs(sampled_seq):
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


def pairs_construction(seqs: List[List[Union[str, int]]], drop_duplicates: bool = True, n_jobs: int = 4):
    """
    Helper function to make pairs from sequences in parallel
    Parameters
    ----------
    seqs : input sequences of nodes
    drop_duplicates : bool, default if True
        Delete pairs where both elements are the same
    n_jobs : int, default is 4
        Number of workers to be created in parallel pool

    Returns
    -------
    List of pairs of nodes as <cur_vertex, context_vertex>
    """
    local_logger = logging.getLogger(f"{__name__}")
    max_processes = max(n_jobs, os.cpu_count())
    pairs_pool = ProcessPool(nodes=max_processes)
    pairs_pool.terminate()
    pairs_pool.restart()
    local_logger.info("Started making pairs from the sequences.")
    pairs = pairs_pool.map(_make_pairs, seqs)
    local_logger.info(f"Total number of raw sampled pairs is {len(pairs)}")
    if drop_duplicates:
        pairs = [item for sublist in pairs for item in sublist if item[0] != item[1]]
    else:
        pairs = [item for sublist in pairs for item in sublist]
    pairs = [item for item in pairs if (item[0] != -3) & (item[1] != -3)]
    pairs_pool.terminate()
    pairs_pool.restart()
    return pairs
