import gym
from math import inf
from matplotlib.pyplot import close

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

        self.action_space = spaces.Discrete(self.data.num_nodes)
        self.observation_space = spaces.Box(low=-inf, high=inf, shape=(1,))  # graph instance
        self.actions_taken = [1 for x in range(self.action_space.n)]
        self.done_list = [-inf for x in range(self.action_space.n)]

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid Action"

        done = False
        info = action

        old_data = self.data
        self.data = self.flip_spin(action)
        reward = compute_energy(old_data) - compute_energy(self.data)

        next_state = self.data

        self.actions_taken[action] = -inf

        if self.actions_taken == self.done_list:
            done = True

        return next_state, reward, done, info

    def reset(self):
        # take graph from dataset
        self.data = generate_ising_lattice(self.dim, spin_conf="all_up")
        # reset action taken
        self.actions_taken = [1 for x in range(self.action_space.n)]

    def render(self, mode='human'):
        close("all")
        plot_graph(self.data)

    def flip_spin(self, spin):
        spins = torch.clone(self.data.x)
        new_data = self.data.clone()
        spins[spin] = torch.tensor([-1.0])
        new_data.x = spins
        return new_data


class IsingGraph2d(IsingGraph2dRandom):
    metadata = {"render.modes": ["human"]}

    def __init__(self, data):
        super(IsingGraph2dRandom, self).__init__()

        self.original_data = data
        self.data = data
        self.action_space = spaces.Discrete(self.data.num_nodes)
        self.observation_space = spaces.Box(low=-inf, high=inf, shape=(1,))  # graph instance
        self.actions_taken = [1 for x in range(self.action_space.n)]
        self.done_list = [-inf for x in range(self.action_space.n)]

    def reset(self):
        self.data = self.original_data
        # reset action taken
        self.actions_taken = [1 for x in range(self.action_space.n)]




