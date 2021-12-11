import torch
import random as rn
import networkx as nx

from torch_geometric.utils import from_networkx, to_undirected
from networkx.generators.lattice import grid_graph
from networkx.algorithms.operators.all import disjoint_union_all
from networkx.classes.function import set_edge_attributes, set_node_attributes
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset, Batch
from collections import defaultdict

class IsingDataset2d(InMemoryDataset):

    def __init__(self, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)


def generate_ising_lattice(dim, distribution="gauss", params=None, spin_conf="all_up", periodic=False, external=True):
    """
    Generates random Ising Graph in lattice format.
    :param dim: tuple of sizes in each dimension
    :param distribution: Decides which distribution is used to generate coupling strength
    :param params: parameters of chosen distribution
    :param spin_conf: decides spin configuration
    :param periodic: periodic or fixed boundary conditions
    :param external: external magnetic field
    :return:
    """
    assert distribution in ["gauss", "uniform"], "distribution must be \"gauss\" or \"uniform\""
    assert spin_conf in ["all_up", "all_down",
                         "random"], "spin configuration must be \"all_up\", \"all_down\" or \"random\""

    x = []
    edge_attr = []

    # create grid and get basic graph attributes
    graph = grid_graph(dim)
    num_of_nodes = graph.number_of_nodes()

    if external:
        edge_list = [(v, v) for v in graph.nodes()]
        graph.add_edges_from(edge_list)

    if periodic:
        print("WARNING: periodic boundary condition not supported yet")

    num_of_edges = graph.size()

    temp = from_networkx(graph)
    edge_index = to_undirected(temp.edge_index)  # it gives predictable order to COO matrix

    # create spin configuration
    if spin_conf == "all_up":
        x = [[1] for v in range(num_of_nodes)]
    elif spin_conf == "all_down":
        x = [[-1] for v in range(num_of_nodes)]
    elif spin_conf == "random":
        x = [rn.choice([[-1], [1]]) for v in range(num_of_nodes)]
    x = torch.tensor(x, dtype=torch.float)

    # create couplings
    if distribution == "gauss":
        if params is None:
            params = [0, 1]
        edge_attr = generate_edge_attr(edge_index.t().tolist(), distribution, params)
    if distribution == "uniform":
        if params is None:
            params = [-2, 2]
        edge_attr = generate_edge_attr(edge_index.t().tolist(), distribution, params)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_attr.resize_((edge_index.size()[1], 1))  # transpose row vector to column vector

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def generate_dataset(size, batch_size):
    data_list = [generate_ising_lattice((3, 3)) for i in range(size)]
    dataset = DataLoader(data_list, batch_size=batch_size, shuffle=True)

    return dataset


def generate_edge_attr(list_of_edges, dist, params):
    buffor = []
    attr = []
    duplicate = False
    for edge in list_of_edges:

        # get random value from distribution
        if dist == "gauss":
            value = rn.gauss(params[0], params[1])
        elif dist == "uniform":
            value = rn.uniform(params[0], params[1])
        else:
            value = rn.gauss(params[0], params[1])  # safety

        for b in buffor:
            if b[0][0] == edge[1] and b[0][1] == edge[0]:
                attr.append(b[1])
                duplicate = True

        if not duplicate:
            attr.append(value)

        buffor.append([edge, value])
        duplicate = False
    return attr


def transform(data, dim):
    """
    deprecated
    It expects square grid
    Transform ising graph into form with grid coordinates instead of spins (see article)
    :param data: Graph
    :param dim: number of dimensions
    :return: graph with node attributes like in article (node grid coordinates)
    """
    graph = data.clone()
    x = graph.num_nodes
    y = x**(1.0/dim)
    size = [int(y) for i in range(dim)]
    g = grid_graph(size)
    x = list(g.nodes())
    graph.x = torch.tensor(x, dtype=torch.float)
    return graph


def transform_batch_square(batch):
    """
    Transform batch of two dimensional square grid graphs.
    :param batch: DataBach object created by from_data_list class method of Batch
    :return:
    """
    list_of_graphs = []
    for i in range(batch.num_graphs):
        graph = batch.get_example(i).clone()
        x = graph.num_nodes
        y = x ** (1.0 / 2)
        size = [int(y), int(y)]
        g = grid_graph(size)
        x = list(g.nodes())
        graph.x = torch.tensor(x, dtype=torch.float)
        list_of_graphs.append(graph)
    return Batch.from_data_list(list_of_graphs)


def generate_chimera(dim, distribution="gauss", params=None, spin_conf="random"):

    assert distribution in ["gauss", "uniform"], "distribution must be \"gauss\" or \"uniform\""
    assert spin_conf in ["all_up", "all_down",
                         "random"], "spin configuration must be \"all_up\", \"all_down\" or \"random\""

    n = dim[0]
    m = dim[1]
    list_of_graphs = [nx.complete_bipartite_graph(4, 4) for x in range(n*m)]
    g = disjoint_union_all(list_of_graphs)

    # I have found these formulas by hand, k is in range four because we have 4x4 complete bipartite graph
    horizontal_edges = [(8*row*m + 8*column + 4 + k, 8*row*m + 8*column + 12 + k) for row in range(n)
                        for column in range(m-1) for k in range(4)]

    vertical_edges = [(8*row*m + 8*column + k, 8*row*m + 8*column + 8*m + k) for row in range(n-1)
                      for column in range(m) for k in range(4)]

    g.add_edges_from(horizontal_edges)
    g.add_edges_from(vertical_edges)

    # External field is represented as self-loops
    external_list = [(v, v) for v in g.nodes()]
    g.add_edges_from(external_list)

    # create couplings
    if distribution == "gauss":
        if params is None:
            params = [0, 1]
        edge_attr = {edge: [rn.gauss(params[0], params[1])] for edge in g.edges}
    if distribution == "uniform":
        if params is None:
            params = [-2, 2]
        edge_attr = {edge: [rn.uniform(params[0], params[1])] for edge in g.edges}

    set_edge_attributes(g, edge_attr, "coupling")

    # create coupling
    if spin_conf == "random":
        spins = {node: rn.choice([[-1.0], [1.0]]) for node in g.nodes}
    if spin_conf == "all_up":
        x = {node: [1.0] for node in g.nodes}
    elif spin_conf == "all_down":
        x = {node: [-1.0] for node in g.nodes}

    set_node_attributes(g, spins, "spin")

    return g


def nx_to_pytorch(graph):
    """
    This function assumes that node attributes are called "spin" and edge attributes are called "coupling"
    :param graph: networkx Graph
    :return: pytorch geometric data
    """

    data = from_networkx(graph, ["spin"], ["coupling"])

    return data
