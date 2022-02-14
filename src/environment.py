import copy

import gym
import torch

import numpy as np
import random as rn
import networkx as nx

from tqdm import tqdm
from math import inf, exp
from gym import spaces
from copy import deepcopy
from numpy.random import default_rng
from itertools import product
from src.utils import compute_energy_nx,  nx_to_pytorch
from src.data_gen import generate_chimera

rng = default_rng()






class Chimera(gym.Env):
    def __init__(self, graph, include_spin=False):
        super(Chimera, self).__init__()

        self.graph = graph
        self.include_spin = include_spin
        self.chimera = deepcopy(self.graph)  # creates copy of input graph.
        self.state = nx_to_pytorch(self.chimera, include_spin=self.include_spin)

        # Make sure types are as they should
        self.state.x = self.state.x.type(torch.float)
        self.state.edge_attr = self.state.edge_attr.type(torch.float)

        # Define actions and ways to track them
        self.action_space = spaces.Discrete(self.chimera.number_of_nodes())
        self.done_counter = 0
        self.available_actions = list(range(self.chimera.number_of_nodes()))
        self.mask = np.ones(self.chimera.number_of_nodes())

    def step(self, action: int):

        assert self.action_space.contains(action), "Invalid Action"
        assert action in self.available_actions
        done = False
        info = action
        old_state = deepcopy(self.chimera)
        self.flip_spin(action)  # here we change state of environment
        self.done_counter += 1

        self.state = nx_to_pytorch(self.chimera, include_spin=self.include_spin)
        self.state.x = self.state.x.type(torch.float)
        self.state.edge_attr = self.state.edge_attr.type(torch.float)

        reward = self.compute_reward(old_state, self.chimera, action)

        if self.done_counter == self.chimera.number_of_nodes():
            done = True

        self.available_actions.remove(action)
        self.mask[action] = -inf
        return self.state, reward, done, info

    def reset(self):
        # reset actions taken
        self.done_counter = 0
        self.available_actions = list(range(self.chimera.number_of_nodes()))
        self.mask = np.ones(self.chimera.number_of_nodes())
        # reset graph
        self.chimera = deepcopy(self.graph)
        self.state = nx_to_pytorch(self.chimera, include_spin=self.include_spin)
        # Make sure types are as they should
        self.state.x = self.state.x.type(torch.float)
        self.state.edge_attr = self.state.edge_attr.type(torch.float)

    def flip_spin(self, action):
        self.chimera.nodes[action]["spin"] *= -1

    def compute_reward(self, old_graph,  new_graph, action: int):
        # Get neighbourhood. They are identical in both graphs
        """
        delta_i = list(new_graph.neighbors(action))
        delta_i.append(action)  # to include node itself

        g_old = old_graph.subgraph(delta_i)
        g_new = new_graph.subgraph(delta_i)

        e_old = compute_energy_nx(g_old)
        e_new = compute_energy_nx(g_new)
        """
        e_old = compute_energy_nx(old_graph)
        e_new = compute_energy_nx(new_graph)
        diff = e_old - e_new
        reward = 2*diff

        return reward

    def energy(self):
        return compute_energy_nx(self.chimera)

    def brute_force(self):
        assert self.chimera.number_of_nodes() <= 16, "Too, big instance. Size of chimera should be less or equal 16"
        graph = copy.deepcopy(self.chimera)
        lst = [list(i) for i in product([1, -1], repeat=graph.number_of_nodes())]
        min_energy = inf
        for conf in lst:
            values = {node: conf[node] for node in graph.nodes}
            nx.set_node_attributes(graph, values, "spin")
            energy = compute_energy_nx(graph)
            if energy < min_energy:
                min_energy = energy
        return min_energy

    def simulated_annealing(self, iter_max: int, temp: float) -> float:
        curr = copy.deepcopy(self.graph)
        best = copy.deepcopy(curr)
        for i in tqdm(range(iter_max)):
            prop = copy.deepcopy(curr)
            t = temp / float(i + 1)
            action = rn.choice(list(prop.nodes))
            prop.nodes[action]["spin"] *= -1
            diff = compute_energy_nx(prop) - compute_energy_nx(curr)
            if compute_energy_nx(prop) < compute_energy_nx(curr):
                curr = copy.deepcopy(prop)
                if compute_energy_nx(curr) < compute_energy_nx(best):
                    best = copy.deepcopy(curr)

            elif exp(-diff / t) > rn.random():
                curr = copy.deepcopy(prop)
        return compute_energy_nx(best)


class RandomChimera(Chimera):
    def __init__(self, n, m, include_spin=False):
        super(RandomChimera, self).__init__(graph=generate_chimera(n, m), include_spin=include_spin)

        self.n = n
        self.m = m
        self.dim = (self.n, self.m)
        self.include_spin = include_spin

    def reset(self, random_dim=False, low=2, high=3):
        # new instance
        rdim = rng.integers(low, high, 2, endpoint=True)
        self.dim = rdim if random_dim else (self.n, self.m)
        self.chimera = generate_chimera(self.dim[0], self.dim[1])
        self.state = nx_to_pytorch(self.chimera, include_spin=self.include_spin)
        # reset actions taken
        self.done_counter = 0
        self.available_actions = list(range(self.chimera.number_of_nodes()))
        self.mask = np.ones(self.chimera.number_of_nodes())
        self.action_space = spaces.Discrete(self.chimera.number_of_nodes())
        # Make sure types are as they should
        self.state.x = self.state.x.type(torch.float)
        self.state.edge_attr = self.state.edge_attr.type(torch.float)






