import unittest
import networkx as nx
from src.utils import gauge_transformation_nx, compute_energy_nx
from src.data_gen import generate_chimera
from src.environment import RandomChimera


class TestEnvironment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = RandomChimera((2, 2))
        cls.g = cls.env.chimera
        cls.transformed = gauge_transformation_nx(cls.g)

    def test_gauge(self):

        self.assertEqual(compute_energy_nx(self.g), compute_energy_nx(self.transformed))

    def test_environment(self):
        self.env.compute_reward(self.g, 1)

if __name__ == '__main__':
    unittest.main()
