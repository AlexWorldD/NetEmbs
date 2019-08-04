# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
CONFIG.py
Created by lex at 2019-03-26.
"""
PRINT_STATUS = True
# TODO here you can select with what kind of data you are going to work: Simulated or Real one
MODE = "SimulatedData"
# Parameters below are used only for creating hierarchical folders' structure for storing results/cache
ROOT_FOLDER = ""
EXPERIMENT = ["", ""]

# /////// Running parameters \\\\\\\
N_JOBS = 8

# /////// Skip-Gram parameters \\\\\\\
EMBD_SIZE = 8
STEPS = 50000
BATCH_SIZE = 256
NEGATIVE_SAMPLES = 512
LOSS_FUNCTION = "NegativeSampling"

# /////// Sampling parameters \\\\\\\
# STEP configuration
STRATEGY = "MetaDiff"
DIRECTION = "COMBI"
PRESSURE = 10
WINDOW_SIZE = 2
WALKS_PER_NODE = 30
WALKS_LENGTH = 10
ALLOW_BACK = False
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
CL_ALGORITHM = "Agglomerative"  # or "KMeans"
NUM_CL_MAX = 11
GLOBAL_FSN = None

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

# Visualisation settings
FIG_SIZE = [20, 10]
context_settings = {"paper_half": dict(context="paper", font_scale=1.5),
                    "paper_full": dict(context="paper", font_scale=1.8),
                    "talk_half": dict(context="paper", font_scale=3.5),
                    "talk_full": dict(context="talk", font_scale=2)}
