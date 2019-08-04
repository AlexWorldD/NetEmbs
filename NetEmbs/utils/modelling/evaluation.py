# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
evaluation.py
Created by lex at 2019-08-04.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from typing import Optional, List


def MAPE(y_true, y_pred) -> float:
    """
    The mean absolute percentage error (MAPE).


    Parameters
    ----------
    y_true : True values
    y_pred : Predicted values

    Returns
    -------
    MAPE score for the given data
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.divide(np.abs((y_true - y_pred)), y_true, out=np.zeros_like(y_true), where=y_true != 0)) * 100


def NRMSD(y_true, y_pred) -> float:
    """
    Normalized by the InterQuartile range Root-mean-square deviation.

    That metric is less sensitive for extreme values in the target variable.

    NRMSD = RMSE / IQR, where IQR=Q_3 - Q-1
    Parameters
    ----------
    y_true : True values
    y_pred : Predicted values

    Returns
    -------
    Normalized by the InterQuartile range Root-mean-square deviation for the given data
    """
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse) / (np.subtract(*np.percentile(y_true, [75, 25])))


def evaluate_model(df: pd.DataFrame, metric: Optional[str] = "RMSE", n_runs: Optional[int] = 5) -> List[float]:
    """
    Evaluation of the model results.

    Run multiple times and take the average for the selected metric.
    Parameters
    ----------
    df : DataFrame
        Input DataFrame with time series. Train linear model on the 80% of data and 20% used for final evaluation.
    metric : str, optional, default is 'RMSE' - Root mean squared error
    n_runs : int, optional, default is 5
        Number of independent runs to ensure the confidence in the obtained results

    Returns
    -------
    List of scores for n_runs.
    """
    scores = list()
    lr = LinearRegression()
    if "amount_X" not in list(df) or "amount_Y" not in list(df):
        raise KeyError(f"Could not find the columns with X and Y in the given dataset... Titles are {list(df)}, while "
                       f"'amount_X' and 'amount_Y' required!")
    for r_s in range(n_runs):
        #         Make new split
        train, test = train_test_split(df, test_size=0.2, random_state=r_s)
        lr.fit(train.iloc[:, 0].values.reshape(-1, 1), train.iloc[:, 1].values.reshape(-1, 1))
        #         print("Coefficients in constructed linear regression model are: :", lr.coef_)
        tax_predicted = lr.predict(test[["amount_X"]])
        if metric == "MSE":
            scores.append(mean_squared_error(test[['amount_Y']], tax_predicted))
        elif metric == "RMSE":
            scores.append(np.sqrt(mean_squared_error(test[['amount_Y']], tax_predicted)))
        elif metric == "NRMSD":
            scores.append(NRMSD(test[['amount_Y']], tax_predicted))
        elif metric == "MAPE":
            scores.append(MAPE(test[['amount_Y']], tax_predicted))
    return scores
