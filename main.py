"""
for now used mainly for testing purposes
"""


import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from data_gen import generate_ising_lattice, transform
from utils import plot_graph
from learn import EdgeCentric, NodeCentric, SGNN, DIRAC

if __name__ == '__main__':
    g = generate_ising_lattice((3, 3), "gauss")
    #g = to_networkx(g)
    #nx.draw(g)
    #plt.show()

    model = EdgeCentric(2, 4, 1, 2)
    model2 = NodeCentric(2, 4, 1, 2)
    #g = transform(g, (2, 2))
    model3 = SGNN()
    model4 = DIRAC()
    print(model4(g))





