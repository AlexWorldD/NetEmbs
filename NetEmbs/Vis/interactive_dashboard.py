from __future__ import print_function

# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
interactive_dashboard.py
Created by lex at 2019-08-03.
"""
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
from ipywidgets import VBox, widgets
from NetEmbs.Vis.helpers import getColors_Markers
from NetEmbs.Vis.draw import descriptor_for_cluster
from typing import Optional, Dict
import pandas as pd
# import cufflinks
#
# cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

# For selection via multiple traces... stupid way.
tmp_p_see = None
indexes = []
tr_nums = 0


def get_table(header: Optional[Dict[str, Optional[str]]] = None):
    """
    Create a Table for showing journal entries in the selected area
    Parameters
    ----------
    header : Dict, columns title to be used as header->format of cell
        Example: {"ID": None, "FA_Name": None, "Credit": '.4f', "Debit": '.4f', "label": None}
    Returns
    -------
    Plotly FigureWidget with Table
    """
    if header is None:
        header = {"ID": None, "FA_Name": None, "Credit": '.4f', "Debit": '.4f', "label": None}
    t = go.FigureWidget([go.Table(
        header=dict(values=list(header.keys()),
                    fill=dict(color='#E5F1DC'),
                    align=['center'] * 5),
        cells=dict(values=[],
                   format=list(header.values()),
                   #                fill = dict(color='white'),
                   align=['center'] * 5))],
        layout=go.Layout(
            title="Journal Entries",
            autosize=True,
            width=1000,
            height=400))
    return t


def interactive_scatter(df: pd.DataFrame, df_info: pd.DataFrame,
                        table: go.FigureWidget, desc: widgets.Label, word_cloud_area: widgets.Output,
                        legend_title: Optional[str] = "label",
                        n_colors: Optional[int] = 10,
                        word_cloud_title: Optional[str] = "FA_Name"):
    """Create FigureWidget with the scatter plot for the given DataFrame"""
    scatter_data = list()
    cmap, mmap = getColors_Markers(keys=df[legend_title].unique(), cm="tab10", n_colors=n_colors,
                                   markers=["circle", "diamond", "square"])
    for name, group in df.groupby(legend_title):
        scatter_data.append(go.Scatter(x=group.x, y=group.y, mode='markers', name=name,
                                       text=group.apply(lambda row: f"ID={row.ID},   GroundTruth={row.GroundTruth}",
                                                        axis=1),
                                       customdata=group.index.to_list(),
                                       marker=dict(color=cmap[name][1],
                                                   symbol=mmap[name])))
    f = go.FigureWidget(data=scatter_data,
                        layout=go.Layout(
                            title=f"t-SNE visualisation with coloring based on {legend_title}",
                            hovermode='closest',
                            autosize=True,
                            width=1000,
                            height=700))

    def printSignature(trace, points, *args):
        if len(points.point_inds) > 0:
            ids = trace.customdata[points.point_inds[0]]
            row = df.iloc[ids]
            desc.value = f"ID={row.ID},   GroundTruth={row.GroundTruth}"

    def selectBP(trace, points, *args):
        if len(points.point_inds) > 0:
            ids = trace.customdata[points.point_inds[0]]
            row = df.iloc[[ids]]
            chosen_bps = df_info.merge(row, on="ID")
            word_cloud_area.clear_output()
            table.data[0].cells.values = [chosen_bps[col] for col in table.data[0].header.values]

    def filterRows(selected_ids):
        row = df.iloc[selected_ids]
        chosen_bps = df_info.merge(row, on="ID")
        return chosen_bps

    def updateTable(chosen_bps):
        table.data[0].cells.values = [chosen_bps[col] for col in table.data[0].header.values]

    def showClouds(chosen_bps):
        word_cloud_area.clear_output()
        with word_cloud_area:
            descriptor_for_cluster(chosen_bps, legend_title, word_cloud_title, sort_mode="freq", n_top=4)

    scatters = f.data
    max_traces = len(scatters)

    def selectBPs(trace, points, selector):
        global indexes
        global tr_nums
        if not points.point_inds:
            pass
        else:
            indexes.extend([trace.customdata[cur_point] for cur_point in points.point_inds])
        tr_nums = tr_nums + 1
        if tr_nums == max_traces:
            selected_data = filterRows(indexes)
            updateTable(selected_data)
            showClouds(selected_data)
            indexes = []
            tr_nums = 0

    # Hover text: ID and GroundTruth
    for scatter in scatters:
        scatter.hoverinfo = 'text'
        scatter.on_hover(printSignature)
        scatter.on_click(selectBP)
        scatter.on_selection(selectBPs)

    # Selection
    return f


def dashboard(df: pd.DataFrame, df_info: pd.DataFrame, n_colors: Optional[int] = 10,
              color_by: Optional[str] = "GroundTruth",
              wordcloud_with: Optional[str] = "FA_Name",
              table_header: Optional[Dict[str, Optional[str]]] = None):
    """
    Construct interactive dashboard with Plotly

    Parameters
    ----------
    df : DataFrame
        Input DataFrame with 'Emb' column
    df_info : DataFrame
        DataFrame with additional info to be used for words cloud construction. E.g. Journal entries.
    n_colors : int, optional, default is 10
        Number of colours in the scatter plot
    color_by : str, default is 'GroundTruth'
        The column to be used for colouring.
    wordcloud_with : str, default is 'FA_Name'
        The column to be used for construct words cloud and Histograms
    table_header : Dict, columns title to be used as header->format of cell
        Example: {"ID": None, "FA_Name": None, "Credit": '.4f', "Debit": '.4f', "label": None}

    Returns
    -------
    Plotly FigureWidget with Scatter plot, Table and area for Descriptors
    """
    # Label text
    description = widgets.Label(
        value=''
    )
    # WordCouds area
    wordCloudsOutput = widgets.Output()

    table = get_table(header=table_header)
    f_scatter = interactive_scatter(df, df_info, table, description, wordCloudsOutput,
                                    legend_title=color_by, n_colors=n_colors, word_cloud_title=wordcloud_with)
    return VBox([description, f_scatter, table, wordCloudsOutput])
