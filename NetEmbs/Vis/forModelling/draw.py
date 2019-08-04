from __future__ import print_function

# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
draw.py
Created by lex at 2019-08-04.
"""

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from typing import Optional, List, Union, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from NetEmbs import CONFIG

init_notebook_mode(connected=True)


def time_series(dfs: Union[List[pd.DataFrame], pd.DataFrame], agg_title: Optional[str] = "Default signals",
                corr_score: Optional[float] = None, anonymous: Optional[bool] = False, save: Optional[bool] = False,
                **kwargs):
    """
    Plot interactive time series.


    Parameters
    ----------
    dfs : Either one DataFrame or the tuple of DataFrames.
        According to the columns 'amount_X' and 'amount_Y' plot the time series.
    agg_title : str, optional
        The title of figure.
    corr_score : float, optional, default is None
        Correlation coefficient to be added to the title.
    anonymous : bool, optional, default is False
        If True, hide the axis.
    save : bool, optional, default is False
        If True, save the figure to file.
    kwargs

    Returns
    -------
    None
    """
    if corr_score is not None:
        agg_title = agg_title + ".     Correlation: " + str(round(corr_score, 3))
    if len(dfs) > 1:
        fig2 = go.Figure(data=[go.Scatter(x=df.index,
                                          y=df.amount,
                                          name=df.name
                                          ) for df in dfs],
                         layout=go.Layout(width=1200,
                                          height=400, showlegend=True, title=go.layout.Title(text=agg_title),
                                          hovermode='closest',
                                          legend=dict(orientation="h", font=dict(size=18), xanchor='center', x=0.5,
                                                      y=-0.1),
                                          font=dict(size=18)))
    else:
        fig2 = go.Figure(data=go.Scatter(x=dfs.index,
                                         y=dfs.amount,
                                         name=dfs.name,
                                         layout=go.Layout(width=1200,
                                                          height=400, showlegend=True,
                                                          title=go.layout.Title(text=agg_title), hovermode='closest')))
    if anonymous:
        fig2.layout.yaxis = go.layout.YAxis(showticklabels=False)
        fig2.layout.xaxis = go.layout.XAxis(showticklabels=False)
    iplot(fig2)
    if save:
        fig2.write_image(f"{kwargs.get('filename') or ''}.pdf")


def regression_plot(df: pd.DataFrame, context: Optional[str] = "paper_full",
                    labels: Optional[Tuple[str, str]] = None,
                    save: Optional[bool] = False,
                    **kwargs):
    """
    Draw regression plot(s) for the given data.

    Parameters
    ----------
    df : DataFrame with 'amount_X' and 'amount_Y' columns
    context : str, optional, default is 'paper_full'
        String name for context leads to minor changes in font-scale
    labels : Tuple of str, optional
        Labels for X_axis and Y_axis of the plots.
    save : bool, optional, default is False
        Save scatter plot to file.
    kwargs

    Returns
    -------

    """
    labels = labels or ["Time series I", "Time series II"]
    dpi = 140
    sns.set_context(**CONFIG.context_settings.get(context))
    cur_plot = sns.lmplot(x="amount_X", y="amount_Y", col="Model", data=df, col_wrap=2, height=5, aspect=1.5,
                          scatter_kws=dict(s=70, alpha=0.75), line_kws=dict(alpha=0.8)) \
        .set_titles(col_template='{col_name}') \
        .set_xlabels(labels[0]) \
        .set_ylabels(labels[1]) \
        .set(xlim=(0, 1), ylim=(0, 1))
    cur_plot.fig.subplots_adjust(hspace=0.3, wspace=0.1)
    if save:
        cur_plot.savefig(f"{kwargs.get('filename') or ''}RegressionPlot.png", bbox_inches="tight", dpi=dpi,
                         pad_inches=0.05)
    plt.show(cur_plot)
