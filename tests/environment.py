import unittest
import torch
import networkx as nx

from itertools import count
from math import inf
from src.utils import gauge_transformation_nx, compute_energy_nx
from src.environment import RandomChimera
from src.DIRAC import DIRAC
from src.train import select_action

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestEnvironment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = RandomChimera((2, 2))
        cls.g = cls.env.chimera.copy()
        cls.transformed = gauge_transformation_nx(cls.g)
        cls.policy_net = DIRAC().to(device)

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
                self.assertEqual(self.env.available_actions, [])
                self.assertEqual(self.env.mask, [-inf for node in self.env.chimera.nodes])
        self.env.reset()
        self.assertNotEqual(self.env.chimera, self.g)
        self.assertEqual(list(nx.get_node_attributes(self.env.chimera, "spin").values()),
                         [[1.0] for node in self.env.chimera.nodes])
        self.assertEqual(self.env.done_counter, 0)

    def test_train(self):
        for t in count():
            # Select and perform an action
            action = select_action(self.env)
            next_state, reward, done, action_taken = self.env.step(action)
            # Store the transition in memory
            #memory.push(state, action, next_state, reward)

            # Move to the next state
            #state = next_state

            # Perform one step of the optimization (on the policy network)
            #optimize_model()
            if done:  # it is done when model performs final spin flip
                break


if __name__ == '__main__':
    unittest.main()
