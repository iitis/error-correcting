import torch
import random as rn

from torch_geometric.utils import from_networkx, to_undirected
from networkx.generators.lattice import grid_graph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


def generate_ising_lattice(dim, distribution="gauss", params=None, spin_conf="all_up", periodic=False, external=True):
    """

    :param dim:
    :param distribution:
    :param params:
    :param spin_conf:
    :param periodic:
    :param external:
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
        print("periodic boundary condition not supported yet")

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
    data_list = [generate_ising_lattice((5, 5)) for i in range(size)]
    dataset = DataLoader(data_list, batch_size=batch_size)

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


def transform(graph, dim):
    x = graph.num_nodes
    y = x**(1.0/dim)
    size = [int(y) for i in range(dim)]
    g = grid_graph(size)
    x = list(g.nodes())
    graph.x = torch.tensor(x, dtype=torch.float)
    return graph
