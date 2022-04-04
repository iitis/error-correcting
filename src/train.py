import torch
import gym
import math
import os
import csv
import pickle
import time
import networkx as nx
import random as rn
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from pathlib import Path
from copy import deepcopy
from src.environment import RandomChimera, Chimera
from src.utils import TransitionMemory, n_step_transition, compute_energy_nx, nx_to_pytorch
from src.DIRAC import DIRAC
from src.data_gen import generate_chimera, generate_chimera_from_csv_dwave
from itertools import count
from tqdm import tqdm
from torch_geometric.data import Batch
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel
from statistics import mean
from math import inf
from collections import deque
from dataclasses import dataclass
from typing import List

@dataclass
class BaseTrainer:
    """Used for initialisation of all parameters"""

    # Cuda devices
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Default paths
    root_dir = Path.cwd().parent
    model_path = root_dir / "models" / "model_C3_v5.pt"
    model_validation_path = root_dir / "models" / "model_C3_v5_val.pt"
    checkpoint_path = root_dir / "models" / "model_checkpoint_test.pt"
    dwave_path = root_dir / "datasets" / "d_wave"

    # Default parameters values
    batch_size: int = 64
    gamma: float = 0.999
    eps_start: float = 1.0
    eps_end: float = 0.05
    num_episodes: int = 60000
    eps_decay: int = int(num_episodes * 0.20)
    include_spin: bool = True
    episode_checkpoint: int = -1
    validation_score: float = inf
    validation_update: int = 10


    # Default network
    policy_net = DIRAC(include_spin=include_spin)
    policy_net = policy_net.to(device)
    #target_net = DIRAC(include_spin=include_spin)
    #target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001, weight_decay=0.002)
    loss_function = nn.MSELoss()

    def load_checkpoint(self, path=None) -> None:
        print("Loading checkpoint")
        checkpoint = torch.load(self.checkpoint_path) if path is None else torch.load(path)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        #self.target_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.episode_checkpoint = checkpoint["episode"]
        self.validation_score = checkpoint['validation_score']

    def load_model(self, path=None) -> None:

        model = torch.load(self.model_path) if path is None else torch.load(path)
        self.policy_net.load_state_dict(model["model_state_dict"])

    def save_checkpoint(self, episode: int, path=None) -> None:
        save_path = self.checkpoint_path if path is None else path
        torch.save({
            'episode': episode,
            'validation_score': self.validation_score,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()}, save_path)

    def save_model(self, episode: int, path=None) -> None:
        save_path = self.model_path if path is None else path
        torch.save({
            'episode': episode,
            'model_state_dict': self.policy_net.state_dict()}, save_path)

    def change_optimiser(self, optimizer: optim.Optimizer) -> None:   # Don't know if it is needed
        self.optimizer = optimizer


class DQNTrainer(BaseTrainer):
    def __init__(self, env: Chimera) -> None:
        super(DQNTrainer, self).__init__()
        # Maybe memory parameters could be user-specified
        self.trajectory = deque([], maxlen=1000)
        self.replay_buffer = TransitionMemory(240000)  # n-step transition, will have aprox. 70 GB size
        self.steps_done: int = 0

        self.env = env
        self.val_env = Chimera(generate_chimera(8, 8), include_spin=self.include_spin)  # to change
        self.q_actions = None
        self.q_values = None

    def set_val_env(self, n: int, m: int) -> None:
        self.val_env = Chimera(generate_chimera(n, m), include_spin=self.include_spin)

    def reset_trajectory(self) -> None:
        self.trajectory = deque([], maxlen=1000)

    def plot_energy_path(self) -> None:
        self.policy_net.eval()
        self.val_env.reset()
        self.compute_q_values(validate=True)
        min_energy: float = self.val_env.energy()
        energy_list: list = [min_energy]
        for _ in count():
            # Select and perform an action
            action = self.select_action_on_policy(validate=True)
            _, _, done, _ = self.val_env.step(action)
            energy = self.val_env.energy()
            energy_list.append(energy)
            if energy < min_energy:
                min_energy = energy
            # it is done when model performs final spin flip
            if done:
                break
        x = np.arange(len(energy_list))
        plt.plot(x, energy_list)
        plt.show()

    def compute_q_values(self, validate: bool = False) -> None:
        state = self.env.state.to(self.device) if not validate else self.val_env.state.to(self.device)

        self.q_values = self.policy_net(state)
        self.q_actions = torch.argsort(self.q_values, descending=True).tolist()

    def select_action_on_policy(self, validate: bool = False) -> int:

        for action in self.q_actions:
            if not validate:
                if action in self.env.available_actions:
                    return action
            else:
                if action in self.val_env.available_actions:
                    return action

    def select_action_epsilon_greedy(self) -> int:
        sample = rn.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)  # epsilon decay

        if sample > eps_threshold:
            return self.select_action_on_policy()
        else:
            return rn.choice(self.env.available_actions)

    def update_replay_buffer(self):
        pass

    def validate(self) -> float:
        self.policy_net.eval()
        self.val_env.reset()
        self.compute_q_values(validate=True)
        min_energy = self.val_env.energy()
        for _ in count():
            # Select and perform an action
            action = self.select_action_on_policy(validate=True)
            _, _, done, _ = self.val_env.step(action)
            energy = self.val_env.energy()
            if energy < min_energy:
                min_energy = energy
            # it is done when model performs final spin flip
            if done:
                break
        return min_energy

    def optimization_step(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        transitions = self.replay_buffer.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = n_step_transition(*zip(*transitions))

        action_batch = torch.tensor(batch.action, device=self.device)
        reward_batch = torch.tensor(batch.reward_n, device=self.device)
        state_batch = Batch.from_data_list(batch.state).to(self.device)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        # Compute loss
        # Reshape tensor from [batch_size] to [batch_size,1]
        action_batch = action_batch[None, :].t()

        # compute state action values for every graph in batch
        state_action_values = self.policy_net(state_batch).reshape((self.batch_size, -1))

        # gather values for chosen actions
        state_action_values = state_action_values.gather(1, action_batch).view(-1)

        loss = self.loss_function(state_action_values, reward_batch)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping for numerical stability (nn.utils.clip_grad_norm_())
        self.optimizer.step()
        return loss

    def fit(self):
        valid_list: list = []
        loss_list: list = []
        min_los = inf
        self.policy_net.train()
        self.load_checkpoint()
        for episode in tqdm(range(self.num_episodes), leave=None, desc="episodes"):

            # Bring steps_done in line with checkpoint
            if episode < self.episode_checkpoint:
                self.steps_done += 1
                continue

            # Reset environment end trajectories
            self.env.reset()
            self.reset_trajectory()
            self.compute_q_values()
            p = rn.random()
            if p < 0.1:
                self.env.simulated_annealing(100,10)
            loss = 0
            # Perform steps of algorithm
            for _ in count():
                # Select and perform an action
                state = self.env.state.to(self.device)
                action = self.select_action_epsilon_greedy()
                _, reward, done, _ = self.env.step(action)

                # Store the transition in memory
                self.trajectory.append([state, action, reward])

                # Perform one step of optimisation
                loss += self.optimization_step()
                # it is done when model performs final spin flip
                if done:
                    break

            # Update memory buffer with n_step rewards
            terminal_state_t = len(self.trajectory)
            reward_n = 0
            for t in range(terminal_state_t-1, -1, -1):
                reward_n = self.gamma * reward_n + self.trajectory[t][2]
                # state, action reward_n
                self.replay_buffer.push(self.trajectory[t][0].to("cpu"), self.trajectory[t][1], reward_n)

            # increase steps done
            self.steps_done += 1

            # Sum loss statistic
            if episode > self.episode_checkpoint + 5:  # to avoid false data from empty replay buffer
                loss /= self.env.chimera.number_of_nodes()
                loss_list.append(loss.item())
                if loss < min_los:
                    self.save_model(episode, self.model_path)

            # Validate
            """
            if episode % self.validation_update == 0:
                validation_score_new = self.validate()
                valid_list.append(validation_score_new)
                # first few hundred episodes are random and don't generalize well
                if validation_score_new < self.validation_score and episode > 500:
                    self.validation_score = validation_score_new
                    self.save_model(episode, self.model_validation_path)
                """
            self.save_checkpoint(episode)
            if episode%100==0:
                x = np.arange(len(loss_list))
                plt.plot(x, loss_list)
                plt.show()


if __name__ == "__main__":

    env = RandomChimera(3, 3, include_spin=True)
    model = DQNTrainer(env)
    model.fit()
