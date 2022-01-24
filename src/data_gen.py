import torch
import pickle
import ast
import random as rn
import networkx as nx
import pandas as pd
import numpy as np
import dwave_networkx as dnx

from networkx.classes.function import set_edge_attributes, set_node_attributes
from numpy.random import default_rng

rng = default_rng()  # This is recomended in numpy documentation


def generate_couplings(graph, distribution="normal", params=None):
    assert distribution in ["normal", "uniform"], "distribution must be \"normal\" or \"uniform\""

    if distribution == "normal":
        if params is None:
            params = [0, 1]
        edge_attr = {edge: rng.normal(params[0], params[1]) for edge in graph.edges}
        external = {node: rng.normal(params[0], params[1]) for node in graph.nodes}
    if distribution == "uniform":
        if params is None:
            params = [-2, 2]
        edge_attr = {edge: rng.uniform(params[0], params[1]) for edge in graph.edges}
        external = {node: rng.uniform(params[0], params[1]) for node in graph.nodes}
    return edge_attr, external


def generate_spin_conf(graph, spin_conf = "random"):
    assert spin_conf in ["all_up", "all_down",
                         "random"], "spin configuration must be \"all_up\", \"all_down\" or \"random\""

    if spin_conf == "random":
        spins = {node: rn.choice([-1.0, 1.0]) for node in graph.nodes}
    if spin_conf == "all_up":
        x = {node: 1.0 for node in graph.nodes}
    elif spin_conf == "all_down":
        x = {node: -1.0 for node in graph.nodes}

    return spins



def generate_chimera(n, m, distribution="normal", params=None, spin_conf="random"):
    """
    :param n: Number of rows in chimera
    :param m: Number of columns in chimera
    :param distribution: Chosen distribution of values
    :param params: parameters of distribution. Should be given as iterable
    :param spin_conf: spin configuration
    :return:
    """

    assert distribution in ["normal", "uniform"], "distribution must be \"normal\" or \"uniform\""
    assert spin_conf in ["all_up", "all_down",
                         "random"], "spin configuration must be \"all_up\", \"all_down\" or \"random\""

    g = dnx.chimera_graph(n, m, 4)

    # create couplings
    coupling, external = generate_couplings(g, distribution=distribution, params=params)

    set_edge_attributes(g, coupling, "coupling")
    set_node_attributes(g, external, "external")

    # create spin conf
    spins = generate_spin_conf(g, spin_conf=spin_conf)

    set_node_attributes(g, spins, "spin")

    return g


def generate_chimera_from_txt(path, spin_conf="random"):
    """
    Generates chimera with random spins from csv file. It asumes square
    :param spin_conf:
    :param path:
    :return:
    """
    chimera_csv = pd.read_csv(path, skiprows=[0], sep=" ", header=None)
    num_of_nodes = max(chimera_csv[0].max(), chimera_csv[1].max())
    num_of_cells = num_of_nodes/8
    dim = num_of_cells ** (1/2)
    n = int(dim)
    m = int(dim)

    g = dnx.chimera_graph(n, m, 4)

    # create spin conf

    spins = generate_spin_conf(g, spin_conf=spin_conf)

    set_node_attributes(g, spins, "spin")

    # create couplings and external magnetic field
    # they must by shifted by 1
    edge_attr = {}
    external = {}
    for index, row in chimera_csv.iterrows():
        if row[0] == row[1]:
            external[row[0]-1] = row[2]
        else:
            edge_attr[(row[0]-1, row[1]-1)] = row[2]

    set_edge_attributes(g, edge_attr, "coupling")
    set_node_attributes(g, external, "external")

    return g


def create_solution_dict(file, save):

    ground = pd.read_csv(file, delimiter=" ", header=None)
    ground.drop(1, inplace=True, axis=1)


    solution_dict = {}
    for index, row in ground.iterrows():

        spins = {i-3: 1 if row[i] == 1 else -1 for i in range(3, len(row)+1)}
        sol = {"ground": row[2], "spins": spins}
        solution_dict["{}".format(index + 1)] = sol

    with open(save, 'wb') as f:
        pickle.dump(solution_dict, f)


def generate_solved_chimera_from_txt(file, solution_dict, number):

    g = generate_chimera_from_txt(path=file)

    spins = solution_dict["{}".format(number)]["spins"]

    set_node_attributes(g, spins, "spin")

    return g

def generate_chimera_from_csv_dwave(path_to_csv, pos):
    df = pd.read_csv(path_to_csv, index_col=0)
    spins = df["sample"][pos]
    spins = ast.literal_eval(spins)

    external = df["h"][pos]
    external = ast.literal_eval(external)

    edge_attr = df["J"][pos]
    edge_attr = ast.literal_eval(edge_attr)
    graph = nx.Graph()
    graph.add_nodes_from(spins.keys())
    graph.add_edges_from(edge_attr.keys())

    nx.set_node_attributes(graph, spins, "spin")
    nx.set_node_attributes(graph, external, "external")

    nx.set_edge_attributes(graph, edge_attr, "coupling")

    ch = dnx.chimera_graph(16,16,4)
    chimera_index = nx.get_node_attributes(ch, "chimera_index")
    nx.set_node_attributes(graph, chimera_index, "chimera_index")
    return graph