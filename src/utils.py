import math
import copy
import torch

import networkx as nx
import matplotlib.pyplot as plt
import random as rn

from torch_geometric.utils import to_networkx, from_networkx
from collections import namedtuple, deque


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
        data = from_networkx(graph, ["spin", "external", "chimera_index"], ["coupling"])
    else:
        data = from_networkx(graph, ["external", "chimera_index"], ["coupling"])
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


n_step_transition = namedtuple('Transition', ('state', 'action', 'reward_n', 'state_n'))


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

