import unittest
from src.train import DQNTrainer
from src.environment import RandomChimera
from torchinfo import summary

class TestOptim(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = RandomChimera(1, 1, include_spin=True)
        cls.model = DQNTrainer(cls.env)

    def test_shapes(self):
        state = self.env.state.to('cuda:0')
        #print(state.edge_attr.view(-1))
        #print(state.x)
        self.model.policy_net.encoder(state.x, state.edge_index, state.edge_attr)

if __name__ == '__main__':
    unittest.main()
