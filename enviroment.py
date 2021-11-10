import gym
import numpy as np
import pandas as pd
from math import inf
from gym import spaces


class IsingGraph2d(gym.Env):  # this package is badly documented, expect lot of hacking
    """Reimplementing Article: Idea is to pass dataset of Ising Graphs and feed them to agent one by one. Agent sees
        q-values of each nodes and chooses which one to flip according to epsilon-greedy strategy.
        Later we may think about something more refined"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, dataset):
        super(IsingGraph2d, self).__init__()

        self.dataset = dataset  # for now dataset is list of pytorch_geometric.data.Data objects (graphs)
        self.data = dataset[0]  # to be overwritten when .reset method is called

        self.action_space = spaces.Discrete(self.data.num_nodes)
        self.observation_space = spaces.Box(low=0, high=inf, shape=(self.data.num_nodes, ))  # unbounded q-values

    def step(self, action):
        pass

    def reset(self):
        # take graph from dataset
        pass

    def render(self, mode='human'):
        pass
