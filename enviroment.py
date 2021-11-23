import gym
import random as rn
import numpy as np
import pandas as pd
from math import inf

import torch
from gym import spaces
from utils import plot_graph, compute_energy
from data_gen import generate_ising_lattice


class IsingGraph2dRandom(gym.Env):  # this package is badly documented, expect lot of hacking
    """Reimplementing Article: Idea is to pass dataset of Ising Graphs and feed them to agent one by one. Agent sees
        q-values of each nodes and chooses which one to flip according to epsilon-greedy strategy.
        Later we may think about something more refined"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, dim):
        super(IsingGraph2dRandom, self).__init__()

        self.dim = dim
        self.data = generate_ising_lattice(dim, spin_conf="all_up")  # to be overwritten when .reset method is called,

        self.state = self.data

        self.action_space = spaces.Discrete(self.data.num_nodes)
        self.observation_space = spaces.Box(low=-inf, high=inf, shape=(self.data.num_nodes, self.data.num_edges,
                                                                       self.data.num_edges, self.data.num_edges))  # graph instance

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"

        done = False

        old_data = self.data
        self.data = self.flip_spin(action)
        reward = compute_energy(old_data) - compute_energy(self.data)
        state = self.data
        info = []

        return state, reward, done, info

    def reset(self):
        # take graph from dataset
        self.data = generate_ising_lattice(self.dim, spin_conf="all_up")

    def render(self, mode='human'):

        plot_graph(self.data)

    def flip_spin(self, spin):
        spins = torch.clone(self.data.x)
        new_data = self.data.clone()
        spins[spin] = torch.tensor([-1.0])
        new_data.x = spins
        return new_data

