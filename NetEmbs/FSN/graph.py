# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
graph.py
Last modified by lex at 2019-03-14.
"""
import networkx as nx
from NetEmbs.CONFIG import LOG
import logging


class FSN(nx.DiGraph):
    """
    Financial Statement Network class. Includes construction, projection, plotting methods
    """

    def __init__(self):
        super().__init__()

    def build(self, df, left_title="Name", right_title="ID"):
        """
        Construct Financial Statement Network (FSN) from DataFrame
        :param df: DataFrame with JournalEntities
        :param left_title: Title of column with FA names: e.g. Name or FA_Name
        :param right_title: Title of column with BP names: e.g. ID, or after clustering Label
        """
        self.add_nodes_from(df[right_title], bipartite=0)
        self.add_nodes_from(df[left_title], bipartite=1)
        self.add_weighted_edges_from(
            [(row[left_title], row[right_title], row["Credit"]) for idx, row in df[df["from"] == True].iterrows()],
            weight='weight', type="CREDIT")
        self.add_weighted_edges_from(
            [(row[right_title], row[left_title], row["Debit"]) for idx, row in df[df["from"] == False].iterrows()],
            weight='weight', type="DEBIT")
        if LOG:
            local_logger = logging.getLogger("NetEmbs.FSN.build")
            local_logger.info("FSN constructed!")
            local_logger.info("Financial accounts are " + str(self.get_BP()))

    def build_default(self):
        """
        Construct Financial Statement Network (FSN) with example Sales-Collection business processes
        :return:
        """
        from NetEmbs.GenerateData.complex_df import sales_collections
        from NetEmbs.DataProcessing.normalize import normalize
        df = normalize(sales_collections())
        self.build(df)

    def get_financial_accounts(self):
        """
        Return the set of Financial Account (FA) nodes in network
        :return: set of Financial Account (FA) nodes
        """
        return [n for n, d in self.nodes(data=True) if d['bipartite'] == 1]

    def get_FA(self):
        """
        Return the set of Financial Account (FA) nodes in network
        :return: set of Financial Account (FA) nodes
        """
        return self.get_financial_accounts()

    def get_business_processes(self):
        """
        Return the set of Business Process (BP) nodes in network
        :return: set of Business Process (BP)  nodes
        """
        return [n for n, d in self.nodes(data=True) if d['bipartite'] == 0]

    def get_BP(self):
        """
        Return the set of Business Process (BP) nodes in network
        :return: set of Business Process (BP)  nodes
        """
        return self.get_business_processes()

    def number_of_BP(self):
        """
        Return total number of Business Process (BP) nodes in network
        :return: integer value
        """
        return len(self.get_BP())

    def projection(self, on="BP"):
        """
        Returns the projection of original FSN onto chosen set of nodes
        :param on: type of nodes to project onto
        :return: Projection
        """
        from networkx.algorithms import bipartite
        if on == "BP":
            project_to = [n for n, d in self.nodes(data=True) if d['bipartite'] == 0]
        elif on == "FA":
            project_to = [n for n, d in self.nodes(data=True) if d['bipartite'] == 1]
        else:
            raise ValueError("Wrong projection argument! {!s} used while FA or BP are allowed!".format(on))
        return bipartite.weighted_projected_graph(self, project_to)
