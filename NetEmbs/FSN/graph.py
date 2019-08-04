# from __future__ import annotations

# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
graph.py
Last modified by lex at 2019-03-14.
"""
import networkx as nx
import logging
import pandas as pd
from typing import Union, List, Dict, Tuple, Optional


class FSN(nx.DiGraph):
    """
    Financial Statement Network class. Includes construction, projection, plotting methods
    """

    def __init__(self):
        super().__init__()
        self.information = dict()

    def build(self, df: pd.DataFrame, left_title: str = "FA_Name", right_title: str = "ID") -> None:
        """
        Construct Financial Statement Network (FSN) from DataFrame

        Use the given DataFrame to construct DiGraph object w.r.t. the Credit/Debit columns
        Parameters
        ----------
        df : DataFrame with JournalEntities
        left_title : str, default is 'FA_Name'
                Title of column with FA names: e.g. Name or FA_Name
        right_title : str, default is 'ID'
                Title of column with BP names: e.g. ID, or after clustering Label

        Returns
        -------
        None
        """
        self.add_nodes_from(df[right_title], bipartite=0)
        self.add_nodes_from(df[left_title], bipartite=1)
        self.add_weighted_edges_from(
            [(row[left_title], row[right_title], row["Credit"]) for idx, row in df[df["flow"] == "outflow"].iterrows()],
            weight='weight', type="CREDIT")
        self.add_weighted_edges_from(
            [(row[right_title], row[left_title], row["Debit"]) for idx, row in df[df["flow"] == "inflow"].iterrows()],
            weight='weight', type="DEBIT")
        self.information = {"left_title": left_title, "right_title": right_title}
        local_logger = logging.getLogger(f"NetEmbs.{__name__}")
        local_logger.info("FSN constructed!")
        local_logger.info(f"Number of Business processes nodes is {self.number_of_BP()}")

    def get_financial_accounts(self) -> List[Union[str, int]]:
        """
        Get the set of Financial Account (FA) nodes in network
        Returns
        -------
        List with FA nodes in FSN
        """
        return [n for n, d in self.nodes(data=True) if d['bipartite'] == 1]

    def get_FAs(self) -> List[Union[str, int]]:
        """
        Get the set of Financial Account (FA) nodes in network
        Returns
        -------
        List with FA nodes in FSN
        """
        return self.get_financial_accounts()

    def get_business_processes(self) -> List[Union[str, int]]:
        """
        Get the set of Business Process (BP) nodes in network
        Returns
        -------
        List with BP nodes in FSN
        """
        return [n for n, d in self.nodes(data=True) if d['bipartite'] == 0]

    def get_BPs(self) -> List[Union[str, int]]:
        """
        Get the set of Business Process (BP) nodes in network
        Returns
        -------
        List with BP nodes in FSN
        """
        return self.get_business_processes()

    def get_credit_flows(self) -> List[Tuple[Union[str, int], Union[str, int]]]:
        return [(u, v) for u, v, d in self.edges(data=True) if d['type'] == "CREDIT"]

    def get_debit_flows(self) -> List[Tuple[Union[str, int], Union[str, int]]]:
        return [(u, v) for u, v, d in self.edges(data=True) if d['type'] == "DEBIT"]

    def number_of_BP(self) -> int:
        """
        Get total number of Business Process (BP) nodes in network
        Returns
        -------
        Int, number of BPs
        """
        return len(self.get_BPs())

    def number_of_FA(self) -> int:
        """
        Get total number of Financial Accounts (FA) nodes in network
        Returns
        -------
        Int, number of FAs
        """
        return len(self.get_FAs())

    def projection(self, on: str = "BP") -> nx.DiGraph:
        """
        Get the projection of original FSN onto chosen set of nodes
        Parameters
        ----------
        on str, default 'BP'
            Type of nodes to get the projection on

        Returns
        -------
        DiGraph object with requested projection
        """
        from networkx.algorithms import bipartite
        if on == "BP":
            project_to = [n for n, d in self.nodes(data=True) if d['bipartite'] == 0]
        elif on == "FA":
            project_to = [n for n, d in self.nodes(data=True) if d['bipartite'] == 1]
        else:
            raise ValueError("Wrong projection argument! {!s} used while FA or BP are allowed!".format(on))
        return bipartite.weighted_projected_graph(self, project_to)

    def info(self) -> Dict:
        """
        Get available info about FSN: number of nodes, total size etc
        Returns
        -------
        Dictionary with available information about FSN
        """
        from NetEmbs.utils.get_size import get_size
        out_info = self.information.copy()
        out_info.update({"BPs": self.number_of_BP(), "FAs": self.number_of_FA(), "Total size": get_size(self)})
        return out_info

    def draw(self, ax: Optional = None) -> None:
        """
        Draw FSN with matplotlib
        Parameters
        ----------
        ax : Matplotlib Axes object, optional, default is None
            Draw the graph in the specified Matplotlib axes
        Returns
        -------
        None
        """
        from NetEmbs.Vis import draw
        draw.fsn(self, ax=ax)
