import torch
import random as rn

from torch_geometric.utils import from_networkx
from networkx.generators.lattice import grid_graph
from torch_geometric.loader import DataLoader


def generate_ising_lattice(dim, distribution="gauss", params=None, spin_conf="all_up", periodic=False, external=True):

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
        edge_attr = [[rn.gauss(params[0], params[1])] for e in range(num_of_edges)]
    if distribution == "uniform":
        if params is None:
            params = [-2, 2]
        edge_attr = [[rn.uniform(params[0], params[1])] for e in range(num_of_edges)]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = from_networkx(graph)
    data.x = x
    data.edge_attr = edge_attr

    return data


def generate_dataset(size, batch_size):
    data_list = [generate_ising_lattice((5, 5)) for i in range(size)]
    dataset = DataLoader(data_list, batch_size=batch_size)
    return dataset