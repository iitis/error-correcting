"""Various useful helper functions"""

import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx


def plot_graph(graph, attributes=True):
    """
    Draws graph. For spin up, node is white, for spin down node is black. Red edge denotes negative
    coupling strength, blue positive.
    :param graph: torch_geometric Data object
    :param attributes: bool, if true then it includes edges and nodes features.
    """
    if attributes:
        node_attrs = graph.x.tolist()
        node_color = []
        for attr in node_attrs:
            if attr == [1.0]:
                node_color.append("white")
            else:
                node_color.append("black")
        edge_attr = graph.edge_attr.tolist()
        edge_attr = [item for sublist in edge_attr for item in sublist]
        edge_attr = set(edge_attr)  # prob that two distinct edges will have the same value is 0
        edge_color = []
        for attr in edge_attr:
            if attr >= 0:
                edge_color.append("blue")
            else:
                edge_color.append("red")
        g = to_networkx(graph, to_undirected=True)
        nx.draw(g, node_color=node_color, edgecolors="black", edge_color=edge_color)

    else:
        g = to_networkx(graph, to_undirected=True)
        nx.draw(g)

    plt.show()


def add_neighbours(x, edge_index): # hacked, but it works
    """
    For each edge it gives sum of its adjacent vertexes. Result is in matrix form corresponding to matrix of
    edge features.
    :param x: matrix of edge features
    :param edge_index: edge_index of graph
    :return: matrix of size [E, num_of_node_features]
    """
    added = []
    for edge in edge_index.t().tolist():
        added.append(x[edge[0]] + x[edge[1]])

    return torch.stack(added)


def add_edges(x, edge_index, edge_attr):  # also hacked, but i don't have better idea
    """

    :param x:
    :param edge_index:
    :param edge_attr: min size [E,2]
    :return:  matrix of size [N, num_of_edge_features]
    """
    added = []
    mask = []
    for node in range(x.size()[0]):
        temp = edge_attr
        for edge in edge_index.t().tolist():
            m = 1 if edge[0] == node else 0
            mask.append([m])
        mask = torch.tensor(mask)
        temp = mask * temp
        added.append(torch.sum(temp, dim=0))
        mask = []
    added = torch.stack(added)
    return added
