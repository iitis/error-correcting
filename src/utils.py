import math
import copy
import torch
import networkx as nx
import matplotlib.pyplot as plt
import random as rn

from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data
from collections import namedtuple, deque

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
        graph.nodes[i]["spin"] = t_list[i] ** 2
    for i, j, data in graph.edges.data():
        data["coupling"] = t_list[i] * t_list[j] * data["coupling"]

    t_node = {node: t_list[node] for node in graph.nodes}
    nx.set_node_attributes(graph, t_node, "t")

    return graph


def compute_energy_nx(nx_graph):
    graph = nx_graph.copy()

    spins = nx.get_node_attributes(graph, "spin")
    external = nx.get_node_attributes(graph, "external")

    s = 0
    for i, j, data in graph.edges(data=True):
        s += data["coupling"] * spins[i] * spins[j]
    for i in spins.keys():
        s += spins[i] * external[i]
    return s


def nx_to_pytorch(graph, include_spin = False):
    """
    This function assumes that node attributes are called "spin" and edge attributes are called "coupling"
    :param graph: networkx Graph
    :return: pytorch geometric data
    """
    if include_spin:
        data = from_networkx(graph, ["spin", "external", "bipartite", "position"], ["coupling"])
    else:
        data = from_networkx(graph, ["external", "bipartite", "position"], ["coupling"])
    return data


def random_spin_flips(nx_graph, percentage):
    """
    :param nx_graph:
    :param percentage:
    :return:
    """
    assert percentage >= 0 and percentage < 1, "percentage should be number between 0 and 1"

    graph = copy.deepcopy(nx_graph)
    nodes = graph.nodes
    number_of_sampled_nodes = math.floor(percentage * graph.number_of_nodes())
    sampled_list = rn.sample(nodes, number_of_sampled_nodes)
    for i in sampled_list:
        graph.nodes[i]["spin"] *= -1
    return graph



n_step_transition = namedtuple('Transition', ('state', 'action', 'reward_n', 'state_n', 'expected'))


class TransitionMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(n_step_transition(*args))

    def sample(self, batch_size):
        return rn.sample(self.memory, batch_size)

    def pop_left(self):
        self.memory.popleft()

    def __len__(self):
        return len(self.memory)

