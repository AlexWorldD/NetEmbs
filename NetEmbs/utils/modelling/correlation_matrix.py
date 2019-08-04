# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
correlation_matrix.py
Created by lex at 2019-08-04.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union


def crosscorr(data_x: pd.Series, data_y: pd.Series, lag: Optional[int] = 0) -> float:
    """
    Calculate Lag-N cross correlation.
    Parameters
    ----------
    data_x : pandas Series
        The first time series
    data_y : pandas Series
        The 2nd time series
    lag : int, optional, default is 0
        Lag in days.

    Returns
    -------
    Cross-correlation with specified lag for the given time series.
    """
    return data_x.corr(data_y.shift(lag, freq="D"))


def calculate_corr(df: pd.DataFrame, query: Tuple[Union[str, int], Union[str, int]],
                   on: Optional[str] = "label", agg_period: Optional[str] = "2D", lag: Optional[int] = 0) -> float:
    """
    Calculate Lag-N cross correlation within the given DataFrame and the sub-set of values in specified column

    Parameters
    ----------
    df : DataFrame
    query : Tuple of selected values in column
    on : str, optional, default is 'label'
        Title of column to be used for filtering
    agg_period : str, optional, default is '2D' (two days)
        Aggregation period for time series
    lag : int, optional, default is 0
        Lag in days.

    Returns
    -------
    Cross-correlation.
    """
    X, Y = [df[df[on] == q] for q in query]
    return crosscorr(X.amount.resample(agg_period).sum(), Y.amount.resample(agg_period).sum(), lag)


def get_corr_matrix(df: pd.DataFrame, on: Optional[str] = "label", agg_period: Optional[str] = "2D",
                    lag: Optional[int] = 0):
    """

    Parameters
    ----------
    df : DataFrame
    on : str, optional, default is 'label'
        Title of column to be used for filtering
    agg_period : str, optional, default is '2D' (two days)
        Aggregation period for time series
    lag : int, optional, default is 0
        Lag in days.

    Returns
    -------

    """
    labels = sorted(df[on].unique())
    corr_matrix = np.zeros((len(labels), len(labels)))
    for x_pos, x in enumerate(labels):
        for y_pos, y in enumerate(labels):
            if x != y:
                try:
                    corr_matrix[x_pos, y_pos] = calculate_corr(df, query=(x, y), on=on, agg_period=agg_period, lag=lag)
                except FloatingPointError:
                    corr_matrix[x_pos, y_pos] = 0.0
    return corr_matrix
