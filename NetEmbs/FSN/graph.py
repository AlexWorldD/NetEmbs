# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
graph.py
Last modified by lex at 2019-03-14.
"""
import networkx as nx


class FSN(nx.DiGraph):
    """
    Financial Statement Network class. Includes construction, projection, plotting methods
    """

    def __init__(self):
        super().__init__()

    def build(self, df):
        """
        Construct Financial Statement Network (FSN) from DataFrame
        :param df: DataFrame with JournalEntities
        """
        self.add_nodes_from(df['ID'], bipartite=0)
        self.add_nodes_from(df['Name'], bipartite=1)
        self.add_weighted_edges_from(
            [(row['Name'], row['ID'], row["Credit"]) for idx, row in df[df["from"] == True].iterrows()],
            weight='weight', type="CREDIT")
        self.add_weighted_edges_from(
            [(row['ID'], row['Name'], row["Debit"]) for idx, row in df[df["from"] == False].iterrows()],
            weight='weight', type="DEBIT")
