# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
CONFIG.py
Created by lex at 2019-03-26.
"""
FIG_SIZE = [20, 10]
PRINT_STATUS = True

# /////// Running parameters \\\\\\\
N_JOBS = 8

# /////// Skip-Gram parameters \\\\\\\
EMBD_SIZE = 16
STEPS = 50000

# /////// Sampling parameters \\\\\\\
# STEP configuration
STEPS_VERSIONS = ["DefUniform", "DefWeighted", "MetaUniform", "MetaWeighted", "MetaDiff"]
PRESSURE = 30
WINDOW_SIZE = 3
WALKS_PER_NODE = 10
WALKS_LENGTH = 10
# Signatures round to decimals
N_DIGITS = 5

# /////// Logging configuration \\\\\\\
MAIN_LOGGER = None
LOG = True

# /////// Clustering configuration \\\\\\\
NUM_CL_MAX = 10
GLOBAL_FSN = None