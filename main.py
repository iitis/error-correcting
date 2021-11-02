"""
for now used mainly for testing purposes
"""


import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.utils import to_networkx
from data_gen import generate_ising_lattice
from utils import plot_graph
from data_gen import generate_dataset


if __name__ == '__main__':
    g = generate_ising_lattice((2, 2), "gauss")
    #g = to_networkx(g)
    #nx.draw(g)
    #plt.show()

    print(g.edge_index)
    print(g.edge_attr)





