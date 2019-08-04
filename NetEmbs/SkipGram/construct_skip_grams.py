# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
construct_skip_grams.py
Created by lex at 2019-08-01.
"""
from NetEmbs.GraphSampling.sampling import graph_sampling, pairs_construction
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
from NetEmbs.utils.get_size import get_size
from tqdm.auto import tqdm
from NetEmbs.utils.update_config import set_new_config


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

    def number_BPs(self):
        return len(self.original_bps)

    def encode(self, original_seq):
        return [self.encoder[item] for item in original_seq]

    def decode(self, seq):
        return [self.decoder[item] for item in seq]

    def encode_pairs(self, original_pairs):
        return [(self.encoder[item[0]], self.encoder[item[1]]) for item in original_pairs]

    def decode_pairs(self, encoded_pairs):
        return [(self.decoder[item[0]], self.decoder[item[1]]) for item in encoded_pairs]


def get_SkipGrams(graph: FSN, use_cache: Optional[bool] = True, **kwargs) \
        -> (List[List[Union[str, int]]], TransformationBPs):
    """
    Helper function to construct Skip-Grams for the given FSN object
    Parameters
    ----------
    graph : FSN
        The graph to be processed
    use_cache : bool, default is True
        To use the previously cached files
    kwargs : additional arguments to be parsed later
    Returns
    -------
    Skip-Grams as list of pairs ready to feed in TF model
        and Transformer Encode/Decode for real values of BPs to integer ones for TF model
    """
    set_new_config(**kwargs)
    tr = TransformationBPs(graph.get_BPs())
    local_logger = logging.getLogger(f"{__name__}")
    if graph is None:
        if CONFIG.GLOBAL_FSN is not None:
            graph = CONFIG.GLOBAL_FSN
        else:
            local_logger.error(f"No given graph object as well as no global one. Stopped execution!")
            raise ValueError(f"No given graph object as well as no global one...")
    if use_cache:
        local_logger.info("Loading SkipGrams from cache... wait...")
        try:
            with open(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + "skip_grams_cached.pkl", "rb") as file:
                skip_gr = pickle.load(file)
            return skip_gr, tr
        except FileNotFoundError:
            local_logger.info("File not found... Recalculate \n")
            pass
        except Exception as e:
            local_logger.error(f"Unexpected error: {e}")
    sampled_seq = graph_sampling(graph, use_cache=use_cache, **kwargs)
    local_logger.info("Start sampling... wait...")
    skip_gr = tr.encode_pairs(pairs_construction(sampled_seq, window_size=kwargs.get("window_size")))
    if use_cache:
        with open(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + "skip_grams_cached.pkl", "wb") as file:
            pickle.dump(skip_gr, file)
    local_logger.info(f"Total number of GOOD sampled pairs is  {len(skip_gr)}")
    return skip_gr, tr
