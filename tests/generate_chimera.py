import unittest
import networkx as nx
import matplotlib.pyplot as plt
from src.data_gen import generate_chimera
from src.utils import nx_to_pytorch


class TestChimeraGeneration(unittest.TestCase):
    dim1 = 3
    dim2 = 3
    draw = False
    print = False

    @classmethod
    def setUpClass(cls):
        cls.g = generate_chimera((cls.dim1, cls.dim2))

    def test_prints(self):
        if self.draw:
            nx.draw(self.g, with_labels=True)
            plt.show()

        if self.print:
            print("nodes: ", self.g.nodes)
            print("edges: ", self.g.edges)
            print("spins: ", nx.get_node_attributes(self.g, "spin"))
            print("couplings: ", nx.get_edge_attributes(self.g, "coupling"))

    def test_nodes(self):
        self.assertEqual(self.g.number_of_nodes(), self.dim1 * self.dim2 * 8)

    def test_degrees(self):
        # degrees with self-loops
        max_degree = max(self.g.degree, key=lambda x: x[1])[1]
        min_degree = min(self.g.degree, key=lambda x: x[1])[1]
        if self.dim1 <= 2 or self.dim2 <= 2:
            self.assertTrue(max_degree == 5)
            self.assertTrue(min_degree == 5)
        else:
            self.assertTrue(max_degree == 6)
            self.assertTrue(min_degree == 5)

    def test_attributes(self):
        self.assertEqual(len(nx.get_node_attributes(self.g, "spin")), len(self.g.nodes))
        self.assertEqual(len(nx.get_edge_attributes(self.g, "coupling")), len(self.g.edges))

    def test_pytorch(self):
        data = nx_to_pytorch(self.g)

        self.assertEqual(self.g.number_of_nodes(), data.num_nodes)
        self.assertEqual(self.g.size(), data.num_edges/2)

        self.assertIsNotNone(data.x)
        self.assertIsNotNone(data.edge_attr)

        self.assertEqual(list(data.x.shape), [data.num_nodes, 5])
        self.assertEqual(list(data.edge_attr.shape), [data.num_edges, 1])


if __name__ == '__main__':
    unittest.main()
