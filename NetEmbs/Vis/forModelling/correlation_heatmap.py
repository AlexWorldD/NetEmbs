# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
correlation_heatmap.py
Created by lex at 2019-08-04.
"""
# For interactive visualization
from plotly.offline import iplot
import numpy as np
import pandas as pd
import plotly.graph_objs as go
# For static visualization
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Union

axis_prefix = {"label": "Cluster ", "GroundTruth": ""}


def make_annotations(z, x, y, annotation_text):
    """
    Helper function to construct annotations array for plotly.

    Returns
    -------
    Plotly layout.Annotation object
    """
    annotations = []
    for n, row in enumerate(z):
        for m, val in enumerate(row):
            annotations.append(
                go.layout.Annotation(
                    text=str(annotation_text[n][m]) if annotation_text[n][m] != 0.0 else "",
                    x=x[m],
                    y=y[n],
                    xref='x1',
                    yref='y1',
                    font=dict(color="Black"),
                    showarrow=False))
    return annotations


def corr_heatmap_interactive(corr_matrix: np.ndarray, labels: List[Union[str, int]], on: Optional[str] = "label"):
    """
    Interactive heatmap with plotly.

    Parameters
    ----------
    corr_matrix : correlation matrix as numpy.ndarray.
    labels : list of labels,
        To be used as X_Axis and Y_Axis labels in the built heatmap.
    on : str, optional, default is 'label'
        String to be used in the title of heatmap.

    Returns
    -------

    """
    x = [axis_prefix[on] + str(cl) for cl in labels]
    y = [axis_prefix[on] + str(cl) for cl in labels]
    # Generate a mask for the upper triangle
    mask = np.ones_like(corr_matrix, dtype=np.bool)
    mask[np.tril_indices_from(mask, k=0)] = False
    corr_matrix[mask] = 0.0
    z_text = np.round(corr_matrix, 2)
    cs = [[00.0, 'rgb(31, 119, 180)'],  # blue
          [0.5, 'rgb(255,255,255)'],  # white
          [1, 'rgb(214, 39, 40)']]  # red

    trace = go.Heatmap(z=corr_matrix, x=x, y=y, zmid=0, zmin=-1, zmax=1, colorscale=cs, showscale=True,
                       colorbar={"thickness": 20, "len": 0.5, "outlinewidth": 0, "xpad": 25,
                                 "title": {"text": "\n \n Correlation", "side": "right"}})
    fig = go.Figure(data=[trace])
    fig.layout.title = go.layout.Title(text="Cross-correlation for the " + on)
    fig.layout.height = len(labels) * 70 + 200
    fig.layout.width = len(labels) * 70 + 25 + 25 + 150
    fig.layout.margin = go.layout.Margin(l=100, r=50, b=100, t=200, pad=10)
    fig.layout.annotations = make_annotations(corr_matrix, x, y, z_text)
    fig.layout.xaxis.side = "top"
    fig.layout.yaxis.automargin = True
    fig.layout.xaxis.automargin = True
    iplot(fig)


def corr_heatmap_static(corr_matrix: np.ndarray, labels: List[Union[str, int]], on="label"):
    """
    Static heatmap with seaborn.

    Parameters
    ----------
    corr_matrix : correlation matrix as numpy.ndarray.
    labels : list of labels,
        To be used as X_Axis and Y_Axis labels in the built heatmap.
    on : str, optional, default is 'label'
        String to be used in the title of heatmap.

    Returns
    -------

    """
    sns.set_context("paper",
                    rc={'figure.figsize': (20, 10), "font.size": 12, "axes.titlesize": 16, "axes.labelsize": 20,
                        "xtick.labelsize": 12, "ytick.labelsize": 12})
    x_ticks = [axis_prefix[on] + str(cl) for cl in labels]
    y_ticks = [axis_prefix[on] + str(cl) for cl in labels]
    z_text = np.round(corr_matrix, 2)
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr_matrix, mask=mask, annot=z_text, cmap="RdBu_r", vmin=-1, vmax=1, center=0, xticklabels=x_ticks,
                yticklabels=y_ticks,
                square=True, linewidths=.1, cbar_kws={"shrink": .5})
    ax.set_title("Cross-correlation for the " + on)
    plt.tight_layout()
    plt.plot()


def correlation_heatmap(df: pd.DataFrame, on: Optional[str] = "label", agg_period: Optional[str] = "2D",
                        lag: Optional[int] = 0, interactive: Optional[bool] = True):
    """
    Construct heatmap for the given DataFrame
    Parameters
    ----------
    df : DataFrame
    on : str, optional, default is 'label'
        Title of column to be used for filtering
    agg_period : str, optional, default is '2D' (two days)
        Aggregation period for time series
    lag : int, optional, default is 0
        Lag in days.
    interactive : bool, optional, default is True
        Use plotly for interactive heatmap.

    Returns
    -------

    """
    from NetEmbs.utils.modelling.correlation_matrix import get_corr_matrix
    corr_matrix = get_corr_matrix(df, on=on, agg_period=agg_period, lag=lag)
    labels = sorted(df[on].unique())
    if interactive:
        #         Use plotly
        return corr_heatmap_interactive(corr_matrix, labels=labels, on=on)
    else:
        #         Use seaborn for visualization
        corr_heatmap_static(corr_matrix, labels=labels, on=on)
