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
    #g = generate_ising_lattice((4, 4, 4), "gauss")
    #g = to_networkx(generate_ising_lattice((4, 4)))
    #nx.draw(g)
    #plt.show()
    dataset = generate_dataset(10, 2)
    for x in dataset:
        print(x)




