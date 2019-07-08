# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
CONFIG.py
Created by lex at 2019-03-26.
"""
FIG_SIZE = [20, 10]
PRINT_STATUS = True
# TODO here you can select with what kind of data you are going to work: Simulated or Real one
MODE = "SimulatedData"
# Parameters below are used only for creating hierarchical folders' structure for storing results/cache
ROOT_FOLDER = ""
EXPERIMENT = ["", ""]

# /////// Running parameters \\\\\\\
N_JOBS = 8

# /////// Skip-Gram parameters \\\\\\\
EMBD_SIZE = 32
STEPS = 100000
BATCH_SIZE = 64
NEGATIVE_SAMPLES = 32

# What save for TensorBoard during model training: "full" includes min/max/mean/std for weights/biases, but very expensive
# LOG_LEVEL = "full"
# "cost" includes only the cost values
LOG_LEVEL = "cost"

# /////// Sampling parameters \\\\\\\
# STEP configuration
STRATEGY = "MetaDiff"
DIRECTION = "COMBI"
PRESSURE = 30
WINDOW_SIZE = 2
WALKS_PER_NODE = 30
WALKS_LENGTH = 2
# Inactive parameters below, under development...
HACK = 0
DOUBLE_NEAREST = False
# Signatures round to decimals
N_DIGITS = 5

all_sampling_strategies = ["OriginalRandomWalk", "DefUniform", "DefWeighted", "MetaUniform", "MetaWeighted", "MetaDiff"]

# /////// Logging configuration \\\\\\\
MAIN_LOGGER = "GlobalLogs"
LOG = True

# /////// Clustering configuration \\\\\\\
NUM_CL_MAX = 11
GLOBAL_FSN = None

# /////// Overwrite default setting for experiments here \\\\\\\\\\
# Last Experiments
EMBD_SIZE = 32
STEPS = 100000
BATCH_SIZE = 64
NEGATIVE_SAMPLES = 32

STRATEGY = "MetaDiff"
DIRECTION = "COMBI"
PRESSURE = 30
WINDOW_SIZE = 2
WALKS_PER_NODE = 20
WALKS_LENGTH = 8

# Current working folder
path_postfix_samplings = "_" + "version" + str(STRATEGY) \
                         + "_direction" + str(DIRECTION) \
                         + "_walks" + str(WALKS_PER_NODE) \
                         + "_pressure" + str(PRESSURE) \
                         + "_1hopFraction" + str(HACK) + "/"

path_postfix_win = "windowSize" + str(WINDOW_SIZE) + "/"
path_postfix_tf = "TFsteps" + str(STEPS) \
                  + "batch" + str(BATCH_SIZE) \
                  + "_emb" + str(EMBD_SIZE) + "/"

WORK_FOLDER = ("", "", "")
