"""
for now used mainly for testing purposes
"""


import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from data_gen import generate_ising_lattice, transform
from enviroment import IsingGraph2d
from utils import plot_graph
from learn import EdgeCentric, NodeCentric, SGNN, DIRAC

if __name__ == '__main__':
    g = generate_ising_lattice((3, 3), "gauss", spin_conf="random")
    #plot_graph(g, True)

    model = EdgeCentric(2, 4, 1, 2)
    model2 = NodeCentric(2, 4, 1, 2)
    model3 = SGNN()
    g = transform(g, 2)
    model4 = DIRAC()
    output = model4(g)
    env = IsingGraph2d([g])

    print(env)





