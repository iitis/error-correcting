import gym
from math import inf
from matplotlib.pyplot import close

import torch
from gym import spaces
from src.utils import plot_graph, compute_energy, gauge_transformation, gauge_transformation_nx, compute_energy_nx,  nx_to_pytorch
from src.data_gen import generate_ising_lattice, generate_chimera


class IsingGraph2dRandom(gym.Env):  # this package is badly documented, expect lot of hacking
    """Reimplementing Article: Idea is to pass dataset of Ising Graphs and feed them to agent one by one. Agent sees
        q-values of each nodes and chooses which one to flip according to epsilon-greedy strategy.
        Later we may think about something more refined"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, dim):
        super(IsingGraph2dRandom, self).__init__()

        self.dim = dim
        self.data = self.generate_graph()  # to be overwritten when .reset method is called,

        self.action_space = spaces.Discrete(self.data.num_nodes)
        self.observation_space = spaces.Box(low=-inf, high=inf, shape=(1,))  # graph instance
        self.actions_taken = [1 for x in range(self.action_space.n)]
        self.done_list = [-inf for x in range(self.action_space.n)]

        self.available_actions = list(range(self.action_space.n))

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid Action"

        done = False
        info = action

        old_data = self.data.clone()
        self.data = self.flip_spin(action)
        reward = compute_energy(old_data) - compute_energy(self.data)

        next_state = self.data

        self.actions_taken[action] = -inf

        if self.actions_taken == self.done_list:
            done = True

        self.available_actions.remove(action)

        return next_state, reward, done, info

    def reset(self):
        # take graph from dataset
        self.data = self.generate_graph()
        # reset action taken
        self.actions_taken = [1 for x in range(self.action_space.n)]
        # reset available_actions
        self.available_actions = list(range(self.action_space.n))

    def render(self, mode='human'):
        close("all")
        plot_graph(self.data)

    def flip_spin(self, spin):
        spins = torch.clone(self.data.x)
        new_data = self.data.clone()
        spins[spin] = torch.tensor([-1.0])
        new_data.x = spins
        return new_data

    def generate_graph(self):
        return gauge_transformation(generate_ising_lattice(self.dim, spin_conf="random"))


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
        self.available_actions = list(range(self.action_space.n))

    def reset(self):
        self.data = self.original_data
        # reset action taken
        self.actions_taken = [1 for x in range(self.action_space.n)]
        self.available_actions = list(range(self.action_space.n))


class RandomChimera(gym.Env):
    def __init__(self, dim, include_spin=False):
        super(RandomChimera, self).__init__()

        self.dim = dim
        self.include_spin = include_spin
        self.chimera = gauge_transformation_nx(generate_chimera(self.dim))  # transformed
        self.state = nx_to_pytorch(self.chimera, include_spin=self.include_spin)

        self.action_space = spaces.Discrete(self.chimera.number_of_nodes())
        self.done_counter = 0
        self.available_actions = list(range(self.chimera.number_of_nodes()))
        self.mask = [1 for node in self.chimera.nodes]

    def step(self, action: int):

        assert self.action_space.contains(action), "Invalid Action"

        done = False
        info = action
        self.chimera = self.flip_spin(action)  # here we change state of environment
        self.done_counter += 1
        self.state = nx_to_pytorch(self.chimera, include_spin=self.include_spin)
        reward = self.compute_reward(self.chimera, action)

        if self.done_counter == self.chimera.number_of_nodes():
            done = True

        self.available_actions.remove(action)
        self.mask[action] = -inf
        return self.state, reward, done, info

    def reset(self):
        # new instance
        self.chimera = gauge_transformation_nx(generate_chimera(self.dim))  # transformed
        self.done_counter = 0
        self.available_actions = list(range(self.chimera.number_of_nodes()))
        self.mask = [1 for node in self.chimera.nodes]
        self.state = nx_to_pytorch(self.chimera, include_spin=self.include_spin)

    def flip_spin(self, action):
        graph = self.chimera
        graph.nodes[action]["spin"] = [-1]
        return graph

    def compute_reward(self, nx_graph, action: int):

        delta_i = list(nx_graph.neighbors(action))
        delta_i.append(action)  # to include node itself
        g = nx_graph.subgraph(delta_i)

        return -2 * compute_energy_nx(g)  # "-2" because we want to to minimize energy


class ComputeChimera(RandomChimera):
    def __init__(self, graph, include_spin=False):
        super(ComputeChimera, self).__init__((1,1))
        self.graph = graph
        self.include_spin = include_spin
        self.chimera = gauge_transformation_nx(self.graph)  # transformed
        self.state = nx_to_pytorch(self.chimera, include_spin=self.include_spin)
        self.state.x = self.state.x.type(torch.float)
        self.state.edge_attr = self.state.edge_attr.type(torch.float)

        self.action_space = spaces.Discrete(self.chimera.number_of_nodes())
        self.done_counter = 0
        self.available_actions = list(range(self.chimera.number_of_nodes()))
        self.mask = [1 for node in self.chimera.nodes]

    def reset(self):
        self.chimera = gauge_transformation_nx(self.graph)
        self.done_counter = 0
        self.available_actions = list(range(self.chimera.number_of_nodes()))
        self.mask = [1 for node in self.chimera.nodes]
        self.state = nx_to_pytorch(self.chimera, include_spin=self.include_spin)
        self.state.x = self.state.x.type(torch.float)
        self.state.edge_attr = self.state.edge_attr.type(torch.float)

    def energy(self):
        return compute_energy_nx(self.chimera)
