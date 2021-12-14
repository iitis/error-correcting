

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


def gauge_transformation_nx(nx_graph):

    graph = nx_graph.copy()  # to avoid making changes in original graph
    t_list = list(nx.get_node_attributes(graph, "spin").values())
    for i in range(graph.number_of_nodes()):
        graph.nodes[i]["spin"] = [t_list[i][0] ** 2]
    for i, j, data in graph.edges.data():
        data["coupling"] = [t_list[i][0] * t_list[j][0] * data["coupling"][0]]

    t_node = {node: t_list[node] for node in graph.nodes}
    nx.set_node_attributes(graph, t_node, "t")

    return graph


def compute_energy_nx(nx_graph):
    graph = nx_graph.copy()

    attr = nx.get_node_attributes(graph, "spin").values()
    external = list(nx.get_node_attributes(graph, "external").values())
    spins = list(attr)
    s = 0
    for i, j, data in graph.edges.data():
        s -= data["coupling"][0] * spins[i][0] * spins[j][0]
    for i in range(len(spins)):
        s -= spins[i][0] * external[i][0]
    return s
