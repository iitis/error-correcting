import unittest

import numpy as np
import torch
import networkx as nx

from itertools import count
from math import inf
from copy import deepcopy
from src.utils import compute_energy_nx
from src.environment import RandomChimera, Chimera
from src.DIRAC import DIRAC


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestEnvironment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = RandomChimera(2, 2)
        cls.g = deepcopy(cls.env.chimera)

        cls.val_env = Chimera(cls.g)
        cls.policy_net = DIRAC().to(device)

    def test_environment(self):

        for i in range(self.g.number_of_nodes()):
            _, _, done, _ = self.env.step(i)
            if i == self.g.number_of_nodes()-1:
                self.assertTrue(done)

                spins_before = list(nx.get_node_attributes(self.g, "spin").values())
                spins_before_reversed = [-1.0 * element for element in spins_before]
                spins_after = list(nx.get_node_attributes(self.env.chimera, "spin").values())
                self.assertEqual(spins_before_reversed, spins_after)

                self.assertEqual(self.env.available_actions, [])

                self.assertEqual(self.env.mask.tolist(), [-inf for node in self.env.chimera.nodes])
        self.env.reset()
        self.assertNotEqual(self.env.chimera, self.g)
        self.assertEqual(self.env.done_counter, 0)

if __name__ == '__main__':
    unittest.main()
