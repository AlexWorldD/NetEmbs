# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
add_time_index.py
Created by lex at 2019-08-04.
"""
import pandas as pd
import numpy as np
from typing import Optional


def addDateTimeIndex(df: pd.DataFrame, sim_time_column: Optional[str] = "Time"):
    if sim_time_column not in df.columns:
        raise KeyError(f"Given column name with simulated time {sim_time_column} is not found in DataFrame!")
    df["SimulatedTime"] = df[sim_time_column]
    df["Time"] = df[sim_time_column].apply(lambda x: np.datetime64('2019-01-01') + np.timedelta64(int(x * 205.7), 'm'))
    return df.set_index("Time")
