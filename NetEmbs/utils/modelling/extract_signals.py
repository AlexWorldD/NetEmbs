# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
extract_signals.py
Created by lex at 2019-08-04.
"""
from NetEmbs.utils.modelling.filter_jounal_entries import filter_data
import pandas as pd
from typing import Optional, List, Dict, Tuple

legend_postfix = {"W": ", weekly", "D": ", daily", "M": ", monthly", "2D": ", 2 days"}


def extract_signals(df_all: pd.DataFrame, shift: Tuple[int, int] = (0, 0),
                    query: Optional[List[Dict]] = None,
                    on: Optional[str] = "label", agg_period: Optional[str] = "W",
                    title: Optional[str] = None, legend: Optional[Tuple[str, str]] = None,
                    scale_data: Optional[bool] = True):
    """
    Extract time series from Journal entries w.r.t. the given query.


    Parameters
    ----------
    df_all : DataFrame
        Journal entries with timedate index AND predicted labels.
    shift : Tuple of int, default is (0,0)
        Shift in days for time series.
    query : List of dictionaries
        Query for filtering: select -> [value1, value2, ..] based on the column 'on'
            _with -> additional filter.
        Example:
            {"select": ["ALL"], "_with": {"FA_Name": "Cash", "flow": "inflow"}}
                                        is equal to
            'Select all transactions where FA_Name=Cash and flow=inflow'
    on : str, optional, default is 'label'
        Title of column to be used for filtering
    agg_period : str, optional, default is 'W' (one week)
        Aggregation period for time series
    title : str, optional, default is None
        Title to be used for plotting as well as file saving.
    legend : Tuple of strings, optional, default is None
        Names for time series to be used instead of df.name attribute
    scale_data : bool, optional, default is True

    Returns
    -------
    DataFrame with requested time-series as 'amount_X' and 'amount_Y' columns.
    """
    from NetEmbs.Vis.forModelling import draw
    # Predicted labels
    agg_title = "Aggregated signals"
    left, right = filter_data(df_all, query=query, on=on)
    #     Make required shifts
    left_agg = left.shift(shift[0], freq="D")
    right_agg = right.shift(shift[1], freq="D")
    #     Aling TimeIndexes for correct aggregation.
    st_date = max(left_agg.index[0], right_agg.index[0])
    end_date = min(left_agg.index[-1], right_agg.index[-1])
    left_agg = left_agg[(left_agg.index >= st_date) & (left_agg.index <= end_date)]
    right_agg = right_agg[(right_agg.index >= st_date) & (right_agg.index <= end_date)]
    #     Makre required aggregation
    left_agg = left_agg.resample(agg_period).agg({"amount": sum})
    right_agg = right_agg.resample(agg_period).apply({"amount": sum})
    if legend is not None:
        left_agg.name = legend[0] + legend_postfix[agg_period]
        right_agg.name = legend[1] + legend_postfix[agg_period]
        agg_title += legend_postfix[agg_period]
    #     Add info about aggregation period to legen texts
    else:
        try:
            left_agg.name = left.name + legend_postfix[agg_period]
            right_agg.name = right.name + legend_postfix[agg_period]
            agg_title += legend_postfix[agg_period]
        except KeyError as e:
            left_agg.name = left.name + ", " + agg_period
            right_agg.name = right.name + ", " + agg_period
            agg_title += ", " + agg_period
    print(f"Correlation for given query and given shifts {shift} is \
          {left_agg.amount.corr(right_agg.amount)}")

    all_data = left_agg.join(right_agg, lsuffix="_X", rsuffix="_Y", how="inner")

    draw.time_series([left_agg, right_agg], agg_title=agg_title, filename=title + "_" + agg_period,
                     corr_score=left_agg.amount.corr(right_agg.amount), anonym=True, save=True)
    if scale_data:
        from sklearn.preprocessing import minmax_scale
        all_data["amount_Y"] = minmax_scale(all_data["amount_Y"])
        all_data["amount_X"] = minmax_scale(all_data["amount_X"])
    return all_data
