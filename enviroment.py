import gym
import random as rn
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
        self.observation_space = spaces.Box(low=-inf, high=inf, shape=(self.data.num_nodes, self.data.num_edges,
                                                                       self.data.num_edges, self.data.num_edges))  # graph instance

    def step(self, action):
        pass

    def reset(self):
        # take graph from dataset
        self.data = rn.choice(self.dataset)

    def render(self, mode='human'):
        pass
