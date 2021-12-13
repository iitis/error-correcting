import unittest
import torch

from src.DIRAC import NodeCentric, EdgeCentric, SGNN
from src.data_gen import generate_chimera, nx_to_pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDIRAC(unittest.TestCase):

    cuda = False
    dim1 = 2
    dim2 = 2
    node_par1 = 2
    node_par2 = 2
    edge_par1 = 2
    edge_par2 = 2

    @classmethod
    def setUpClass(cls):
        cls.g = generate_chimera((cls.dim1, cls.dim2))
        cls.data = nx_to_pytorch(cls.g)

        cls.node = NodeCentric(3, cls.node_par1, 1, cls.node_par2)
        cls.edge = EdgeCentric(3, cls.edge_par1, 1, cls.edge_par2)
        cls.sgnn = SGNN()

        if cls.cuda:
            cls.data.cuda()
            cls.node.cuda()
            cls.edge.cuda()
            cls.sgnn.cuda()

        cls.node_sol = cls.node(cls.data.x, cls.data.edge_index, cls.data.edge_attr)
        cls.edge_sol = cls.edge(cls.data.x, cls.data.edge_index, cls.data.edge_attr)
        cls.encode = cls.sgnn(cls.data.x, cls.data.edge_index, cls.data.edge_attr)

    def test_node(self):
        self.assertEqual(list(self.node_sol.shape), [self.data.num_nodes, self.node_par1 + self.node_par2])

    def test_edge(self):
        self.assertEqual(list(self.edge_sol.shape), [self.data.num_edges, self.edge_par1 + self.edge_par2])

    def test_SGNN(self):
        self.assertEqual(list(self.encode.shape), [self.data.num_nodes, 12])


if __name__ == '__main__':
    unittest.main()
