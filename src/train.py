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
    world_size: int = torch.cuda.device_count()

    # Default paths
    root_dir = Path.cwd().parent
    model_path = root_dir / "models" / "model_C3_v3.pt"
    checkpoint_path = root_dir / "models" / "model_checkpoint.pt"
    dwave_path = root_dir / "datasets" / "d_wave"

    # Default parameters values
    batch_size: int = 64
    gamma: float = 0.999
    eps_start: float = 1.0
    eps_end: float = 0.05
    num_episodes: int = 2000
    eps_decay: int = int(num_episodes * 0.20)
    include_spin: bool = True
    episode_checkpoint: int = -1
    validation_score: float = inf


    # Default network
    policy_net = DIRAC(include_spin=include_spin)
    target_net = DIRAC(include_spin=include_spin)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.RMSprop(policy_net.parameters())

    # Device setup
    policy_net = policy_net.to(device)
    target_net = target_net.to(device)
    target_net.eval()

    def load_checkpoint(self, path = None) -> None:
        print("Loading checkpoint")
        checkpoint = torch.load(self.checkpoint_path) if path is None else torch.load(path)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.episode_checkpoint = checkpoint["episode"]
        self.validation_score = checkpoint['validation_score']

    def save_checkpoint(self, path = None) -> None:
        save_path = self.checkpoint_path if path is None else path
        torch.save({
            'episode': self.episode,
            'validation_score': self.validation_score,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()}, save_path)

    def change_optimiser(self, optimizer: optim.Optimizer) -> None: # Don't know if it is needed
        self.optimizer = optimizer


class DQNTrainer(BaseTrainer):
    def __init__(self, env: Chimera):
        super(DQNTrainer, self).__init__()
        # Maybe memory parameters could be user-specified
        self.trajectory = deque([], maxlen=1000)
        self.replay_buffer = TransitionMemory(240000)  # n-step transition, will have aprox. 70 GB size
        self.steps_done: int = 0

        self.env = env
        self.q_actions = None

    def reset_trajectory(self) -> None:
        self.trajectory = deque([], maxlen=1000)

    def compute_q_values(self) -> None:
        state = self.env.state.to(self.device)
        with torch.no_grad():
            # Here choice really depends on INCLUDE_SPIN, if True then q_values_global is None
            q_values = self.policy_net(state)
            self.q_actions = torch.argsort(q_values, descending=True).tolist()


    def select_action_policy(self) -> int:

        for action in self.q_actions:
            if action in self.env.available_actions:
                return action

        if not self.env.available_actions:
            raise ValueError("No more available actions")

    def select_action_epsilon_greedy(self) -> int:
        sample = rn.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)  # epsilon decay

        if sample > eps_threshold:
            return self.select_action_policy()
        else:
            return rn.choice(self.env.available_actions)

    def collect_trajectories(self) -> None:
        for _ in count():
            # Select and perform an action
            state = self.env.state.to(self.device)
            action = self.select_action_epsilon_greedy()
            _, reward, done, _ = self.env.step(action)

            # Store the transition in memory
            self.trajectory.append([state, action, reward])
            if done:  # it is done when model performs final spin flip
                break

    def validate(self) -> float:
        return 0.0
"""

def generate_val_set(num):
    elements = []
    for _ in range(num):
        graph = generate_chimera(16, 16)
        elements.append(graph)
    return elements

def generate_val_set_hard(path, num):
    elements = []
    for i in range(1, num+1):
        name = f"2048_{i}.csv"
        for j in [1,90]:
            graph = generate_chimera_from_csv_dwave(D_WAVE_PATH+name, j)
            elements.append(graph)
    return elements

def select_action_epsilon_greedy(environment, steps_done):

    sample = rn.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)  # epsilon decay

    if sample > eps_threshold:
        return select_action_policy(environment, q_values_global)
    else:
        return rn.choice(environment.available_actions)


def select_action_policy(environment, q_values_global):

    state = environment.state.to(device)

    with torch.no_grad():
        # Here choice really depends on INCLUDE_SPIN, if True then q_values_global is None
        q_values = policy_net(state) if q_values_global is None else q_values_global

        mask = torch.tensor(environment.mask, device=device)  # used to mask available actions
        q_values = torch.add(q_values, 1E-8)  # to avoid 0 * -inf
        action = mask * q_values
        return action.argmax().item()


def validate(val_set):
    s = 0
    for element in val_set:
        state = nx_to_pytorch(element, True).to(device)
        s += policy_net(state).max().item()
    return s/len(val_set)



def optimize_model(t_max):
    if len(memory) < BATCH_SIZE:
        return inf
    sum_loss = 0

    for _ in range(t_max):
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = n_step_transition(*zip(*transitions))

        action_batch = torch.tensor(batch.action, device=device)
        reward_batch = torch.tensor(batch.reward_n, device=device)
        state_batch = Batch.from_data_list(batch.state).to(device)
        stop_states = Batch.from_data_list(batch.state_n).to(device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        # Compute loss
        expected_state_action_values = target_net(stop_states).max() * GAMMA + reward_batch
        state_action_values = policy_net(state_batch).gather(0, action_batch)

        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values)
        sum_loss += loss
        # Optimize the model

        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # gradient clipping for numerical stability
        optimizer.step()

    return sum_loss


if __name__ == "__main__":

    val_set = checkpoint['val_set'] if CHECKPOINT else generate_val_set(10)
    val_set_hard = checkpoint['val_set_hard'] if CHECKPOINT else generate_val_set_hard(D_WAVE_PATH, 5)

    val_q_list= []
    val_hard_q_list = []

    env = RandomChimera(3, 3, include_spin=INCLUDE_SPIN)

    for episode in tqdm(range(NUM_EPISODES), leave=None, desc="episodes"):

        if episode < episode_checkpoint:
            steps_done += 1
            continue

        # Initialize the environment and state
        env.reset()
        trajectory = deque([], maxlen=1000)

        # Initialize if we include spin or not
        if INCLUDE_SPIN:
            q_values_global = None
        else:
            with torch.no_grad():
                q_values_global = policy_net(env.state.to(device))

        # Perform actions
        for t in count():
            # Select and perform an action
            state = env.state.to(device)
            action = select_action_epsilon_greedy(env, steps_done)
            _, reward, done, _ = env.step(action)

            # Store the transition in memory
            trajectory.append([state, action, reward])

            if done:  # it is done when model performs final spin flip
                break
        # Get n-step sum of rewards and and predicted rewards

        # Perform one step of the optimization (on the policy network)
        sum_loss = optimize_model(env.action_space.n)  # one iteration for every move


        for t in range(len(trajectory)):
            # state, action reward_n, state_n
            stop = t + N if len(trajectory) - t > N else len(trajectory)-1
            reward_n = 0
            for k in range(t, stop + 1):
                reward_n += GAMMA * trajectory[k][2]
            memory.push(trajectory[t][0].to("cpu"), trajectory[t][1], reward_n, trajectory[stop][0].to("cpu"))
        steps_done += 1

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            val_q = validate(val_set)
            val_q_list.append(val_q)
            val_hard_q = validate(val_set_hard)
            val_hard_q_list.append(val_hard_q)

        if episode % 1000 == 0:
            x =  np.arange((episode/TARGET_UPDATE) + 1)
            y1 = val_q_list
            y2 = val_hard_q_list
            plt.plot(x, y1, "-b", label="normal")
            plt.plot(x, y2, "-r", label="hard")
            plt.legend(loc="upper left")
            plt.show()

        if sum_loss < validation_score:
            validation_score = sum_loss
            print(validation_score)
            torch.save({
                'episode': episode,
                'model_state_dict': policy_net.state_dict()}, MODEL_PATH)


"""