"""
for now used mainly for testing purposes
"""


import networkx as nx
import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from data_gen import generate_ising_lattice, transform, transform_batch_square
from enviroment import IsingGraph2dRandom
from utils import plot_graph
from learn import EdgeCentric, NodeCentric, SGNN, DIRAC
from torch_geometric.data import Batch, Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    g1 = generate_ising_lattice((3, 3), "gauss", spin_conf="random")
    g2 = generate_ising_lattice((3, 3), "gauss", spin_conf="random")
    g3 = generate_ising_lattice((3, 3), "gauss", spin_conf="random")
    batch = Batch.from_data_list([g1, g2, g3])
    model = DIRAC().to(device)
    batch2 = transform_batch_square(batch)
    print(model(batch2).shape)





