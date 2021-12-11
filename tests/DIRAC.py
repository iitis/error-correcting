import unittest
import torch

from src.DIRAC import NodeCentric
from src.data_gen import generate_chimera, nx_to_pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyTestCase(unittest.TestCase):

    dim1 = 3
    dim2 = 3
    draw = False
    print = False

    @classmethod
    def setUpClass(cls):
        cls.g = generate_chimera((cls.dim1, cls.dim2))
        cls.data = nx_to_pytorch(cls.g)
        cls.data.cuda()

        cls.model = NodeCentric(1, 3, 1, 4)
        cls.model.cuda()

    def test_shape(self):
        print(self.data.edge_attr)


if __name__ == '__main__':
    unittest.main()
