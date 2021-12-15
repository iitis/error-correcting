import unittest
import networkx as nx
from src.utils import gauge_transformation_nx, compute_energy_nx
from src.data_gen import generate_chimera
from src.environment import RandomChimera


class TestEnvironment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = RandomChimera((2, 2))
        cls.g = cls.env.chimera.copy()
        cls.transformed = gauge_transformation_nx(cls.g)

    def test_gauge(self):
        pass
        self.assertEqual(compute_energy_nx(self.g), compute_energy_nx(self.transformed))

    def test_environment(self):

        for i in range(self.g.number_of_nodes()):
            _, _, done, _ = self.env.step(i)
            if i == self.g.number_of_nodes():
                self.assertTrue(done)
                self.assertEqual(list(nx.get_node_attributes(self.env.chimera, "spin").values()),
                                 [[-1.0] for node in self.env.chimera.nodes])
        self.env.reset()
        self.assertNotEqual(self.env.chimera, self.g)
        self.assertEqual(list(nx.get_node_attributes(self.env.chimera, "spin").values()),
                         [[1.0] for node in self.env.chimera.nodes])
        self.assertEqual(self.env.done_counter, 0)

    def test_train(self):
        pass

if __name__ == '__main__':
    unittest.main()
