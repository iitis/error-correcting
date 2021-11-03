import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx


def plot_graph(graph, attributes=False):
    """
    Draws graph
    :param graph: torch_geometric Data object
    :param attributes: bool, if true then it includes edges and nodes features. Not suported now
    """
    if attributes:  # broken
        node_attrs = [str(x) for x in graph.x.tolist()]
        g = to_networkx(graph, node_attrs=node_attrs, to_undirected=True)
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
    :param edge_attr:
    :return:  matrix of size [N, num_of_edge_features]
    """
    added = []
    mask = []
    for node in range(x.size()[0]):
        temp = edge_attr
        for edge in edge_index.t().tolist():
            m = 1 if edge[0] == node else 0
            mask.append(m)
        mask = torch.tensor(mask)
        mask.resize_((edge_index.size()[1], 1))  # transposition of 1d tensor
        temp = mask * temp
        added.append(torch.sum(temp))
        mask = []
    added = torch.stack(added)

    if len(added.size()) == 1:  # 1d tensor
        added.resize_(x.size()[0], 1)
    return added
