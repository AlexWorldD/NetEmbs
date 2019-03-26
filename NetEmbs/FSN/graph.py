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

    def build(self, df, name_column="Name"):
        """
        Construct Financial Statement Network (FSN) from DataFrame
        :param df: DataFrame with JournalEntities
        :param name_column: Title of column with FA names: Name or FA_Name
        """

        self.add_nodes_from(df['ID'], bipartite=0)
        self.add_nodes_from(df[name_column], bipartite=1)
        self.add_weighted_edges_from(
            [(row[name_column], row['ID'], row["Credit"]) for idx, row in df[df["from"] == True].iterrows()],
            weight='weight', type="CREDIT")
        self.add_weighted_edges_from(
            [(row['ID'], row[name_column], row["Debit"]) for idx, row in df[df["from"] == False].iterrows()],
            weight='weight', type="DEBIT")

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
        return bipartite.weighted_projected_graph(self, project_to)
