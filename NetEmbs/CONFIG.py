# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
CONFIG.py
Created by lex at 2019-03-26.
"""
FIG_SIZE = [20, 10]
PRINT_STATUS = True
WORK_FOLDER = "2"
# TODO here you can select with what kind of data you are going to work: Simulated or Real one
MODE = "SimulatedData"

# /////// Running parameters \\\\\\\
N_JOBS = 8

# /////// Skip-Gram parameters \\\\\\\
EMBD_SIZE = 32
STEPS = 200000
BATCH_SIZE = 64
NEGATIVE_SAMPLES = 32
# What save for TensorBoard during model training: "full" includes min/max/mean/std for weights/biases, but very expensive
# LOG_LEVEL = "full"
# "cost" includes only the cost values
LOG_LEVEL = "cost"

# /////// Sampling parameters \\\\\\\
# STEP configuration
STEPS_VERSIONS = ["DefUniform", "DefWeighted", "MetaUniform", "MetaWeighted", "MetaDiff"]
PRESSURE = 30
WINDOW_SIZE = 3
DOUBLE_NEAREST = False
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

# Current working folder
path_postfix_samplings = "_" + "walks" + str(WALKS_PER_NODE) \
                         + "_pressure" + str(PRESSURE) \
                         + "_window" + str(WINDOW_SIZE) + "/"
path_postfix_tf = "TFsteps" + str(STEPS) \
                  + "batch" + str(BATCH_SIZE) \
                  + "_emb" + str(EMBD_SIZE) + "/"
WORK_FOLDER = (WORK_FOLDER + path_postfix_samplings, path_postfix_tf)
