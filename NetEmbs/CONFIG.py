# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
CONFIG.py
Created by lex at 2019-03-26.
"""
# /////// Running parameters \\\\\\\
N_JOBS = 4

# /////// Skip-Gram parameters \\\\\\\
EMBD_SIZE = 4

# /////// Sampling parameters \\\\\\\
# STEP configuration
STEPS_VERSIONS = ["DefUniform", "DefWeighted", "MetaUniform", "MetaWeighted", "MetaDiff"]
PRESSURE = 30
# Signatures round to decimals
N_DIGITS = 5


# /////// Logging configuration \\\\\\\
MAIN_LOGGER = None
LOG = True

