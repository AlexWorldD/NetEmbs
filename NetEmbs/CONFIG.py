# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
CONFIG.py
Created by lex at 2019-03-26.
"""
FIG_SIZE = [20, 10]
PRINT_STATUS = True
WORK_FOLDER = "May10/"

# /////// Running parameters \\\\\\\
N_JOBS = 8

# /////// Skip-Gram parameters \\\\\\\
EMBD_SIZE = 32
STEPS = 10000
BATCH_SIZE = 128

# /////// Sampling parameters \\\\\\\
# STEP configuration
STEPS_VERSIONS = ["DefUniform", "DefWeighted", "MetaUniform", "MetaWeighted", "MetaDiff"]
PRESSURE = 30
WINDOW_SIZE = 4
DOUBLE_NEAREST = True
WALKS_PER_NODE = 30
WALKS_LENGTH = 50
# Signatures round to decimals
N_DIGITS = 5

# /////// Logging configuration \\\\\\\
MAIN_LOGGER = None
LOG = True

# /////// Clustering configuration \\\\\\\
NUM_CL_MAX = 10
GLOBAL_FSN = None