# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
__init__.py.py
Created by lex at 2019-04-12.
"""
# from NetEmbs.SkipGram.tensor_flow import get_embs_TF, add_ground_truth
from NetEmbs.SkipGram.tf_model import get_embeddings
from NetEmbs.SkipGram.construct_skip_grams import get_SkipGrams, TransformationBPs
