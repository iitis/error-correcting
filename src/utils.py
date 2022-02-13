import math
import copy
import torch

import networkx as nx
import matplotlib.pyplot as plt
import random as rn

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
from collections import namedtuple, deque


def plot_graph(graph: nx.Graph, attributes: bool = True) -> None:
    """
    Draws graph. For spin up, node is white, for spin down node is black. Red edge denotes negative
    coupling strength, blue positive.
    :param graph: torch_geometric Data object
    :param attributes: bool, if true then it includes edges and nodes features.
    """
    raise NotImplementedError()


def compute_energy_nx(graph: nx.Graph) -> float:
    g = copy.deepcopy(graph)

    spins = nx.get_node_attributes(g, "spin")
    external = nx.get_node_attributes(g, "external")
    couplings = nx.get_edge_attributes(g, "coupling")

    s = 0
    for i, j in couplings:
        s += couplings[(i, j)] * spins[i] * spins[j]
    for i in external:
        s += external[i] * spins[i]
    return s


def nx_to_pytorch(graph: nx.Graph, include_spin: bool = False) -> Data:
    """
    This function assumes that node attributes are called "spin" and edge attributes are called "coupling"
    :param include_spin:
    :param graph: networkx Graph
    :return: pytorch geometric data
    """
    if include_spin:
        data = from_networkx(graph, ["spin", "external", "chimera_index"], ["coupling"])
    else:
        data = from_networkx(graph, ["external", "chimera_index"], ["coupling"])
    return data


def random_spin_flips(nx_graph: nx.Graph, percentage: float) -> nx.Graph:
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


n_step_transition = namedtuple('Transition', ('state', 'action', 'reward_n'))


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

