"""Various useful helper functions"""

import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    For each vertex it adds edge features of adjacent edges
    :param x: matrix of edge features
    :param edge_index: edge_index of graph
    :param edge_attr: min size [E,2]
    :return:  matrix of size [N, num_of_edge_features]
    """
    added = []
    for node in range(x.size()[0]):
        adjacent_edges = []
        sum_of_edges = torch.zeros(edge_attr.size()[1]).to(device)
        for index, edge in enumerate(edge_index.t().tolist()):
            if edge[0] == node:
                adjacent_edges.append(index)
        for index in adjacent_edges:
            sum_of_edges += edge_attr[index]

        added.append(sum_of_edges)

    added = torch.stack(added)
    return added


def compute_energy(data):
    """
    computes energy of ising spin-glass instance
    :param data: ising spin glass instance
    :return: Energy of the instance
    """
    spins = data.x
    interactions = data.edge_attr
    edge_list = data.edge_index.t().tolist()
    unique_edges = []

    # determine unique edges (data object is directed graph made undirected)
    # index is needed to determine value
    for index, edge in enumerate(edge_list):
        duplicate = False
        for unique in unique_edges:
            if edge[0] == unique[0][1] and edge[1] == unique[0][0]:
                duplicate = True
        if not duplicate:
            unique_edges.append([edge, index])

    # Compute energy. External magnetic field is included in self loops, so we can just sum over all unique edges
    energies = [spins[unique[0][0]] * interactions[unique[1]] * spins[unique[0][1]] for unique in unique_edges]

    energy = -1 * sum(energies).item()

    return energy


def gauge_transformation(data):
    """
    changing the input data of the graph with all the same spin glass
    :param data: ising spin glass instance
    :return: graph with all positive data
    """
    graph = data.clone()
    new_data = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
    edge_list = data.edge_index.t().tolist()
    t = graph.x.clone()
    for i, x in enumerate(new_data.x):
        new_data.x[i] = x * t[i]
    for i, edge in enumerate(edge_list):
        new_data.edge_attr[i] = graph.edge_attr[i] * t[edge[0]] * t[edge[1]]

    return new_data



