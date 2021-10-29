import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx


def plot_graph(graph, attributes=False):
    """
    Draws graph
    :param graph: torch_geometric Data object
    :param attributes: bool, if true then it includes edges and nodes features. Not suported now
    """
    if attributes:  # broken
        node_attrs = [str(x) for x in graph.x.tolist()]
        g = to_networkx(graph, node_attrs=node_attrs, to_undirected=True)
    else:
        g = to_networkx(graph, to_undirected=True)

    nx.draw(g)
    plt.show()
