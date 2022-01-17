import torch
import math
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
from src.utils import TransitionMemory, n_step_transition
from src.DIRAC import DIRAC
from src.data_gen import generate_chimera
from itertools import count
from tqdm import tqdm
from torch_geometric.data import Batch
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel
from statistics import mean
from math import inf
from collections import deque


# Cuda devices
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
world_size = torch.cuda.device_count()
print('Let\'s use', world_size, 'GPUs!')

# Global constants
PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/models/model_C1_C3.pt"
CHECKPOINT_PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/models/model_checkpoint.pt"
VAL_PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/chimera_512.pkl"
DATA_PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/random_C2_trajectory1.pkl"
BEST_VALUES_PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/chimera_512_checkpoint.pkl"

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.05
NUM_EPISODES = 51000
EPS_DECAY = int(NUM_EPISODES * 0.20)
TARGET_UPDATE = 10
N = 30
CHECKPOINT = True
INCLUDE_SPIN = True

# Models and optimizer
policy_net = DIRAC(include_spin=INCLUDE_SPIN)
target_net = DIRAC(include_spin=INCLUDE_SPIN)
target_net.load_state_dict(policy_net.state_dict())
policy_net = policy_net.to(device)
target_net = target_net.to(device)
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())

if CHECKPOINT:
    print("Loading checkpoint")
    checkpoint = torch.load(CHECKPOINT_PATH)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    target_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    episode_checkpoint = checkpoint["episode"]
    val_instance = checkpoint['val_instance']
    validation_score = -52.401285609054625

else:
    episode_checkpoint = -1
    val_instance = generate_chimera(8, 8)
    validation_score = inf

# Global variables
steps_done = 0

memory = TransitionMemory(240000)  # n-step transition, will have aprox. 70 GB size
q_values_global = None


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


def validate(validation_score, val_instance, q_values_global):

    val_env = Chimera(val_instance, include_spin=INCLUDE_SPIN)
    min_eng = validation_score
    energy_path = []
    for _ in count():
        # Select and perform an action
        action = select_action_policy(val_env, q_values_global)
        _, _, done, _ = val_env.step(action)
        energy = val_env.energy()
        energy_path.append(energy)
        if energy < min_eng:
            min_eng = energy

        if done:  # it is done when model performs final spin flip
            break

    x = np.arange(val_env.chimera.number_of_nodes())
    min_eng_plot = [min_eng for _ in range(val_env.chimera.number_of_nodes())]
    plt.plot(x, energy_path, x, min_eng_plot)
    plt.xlabel("steps")
    plt.ylabel("energy")
    plt.title('instance 1, try {}, {}'.format(1, "val"))
    plt.show()

    return min_eng


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return 0, 0, 0, 0

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = n_step_transition(*zip(*transitions))

    action_batch = torch.tensor(batch.action, device=device)
    reward_batch = torch.tensor(batch.reward_n, device=device)
    state_batch = Batch.from_data_list(batch.state).to(device)
    stop_states = Batch.from_data_list(batch.state_n).to(device)

    expected_state_action_values = target_net(stop_states).max() * GAMMA + reward_batch

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    state_action_values = policy_net(state_batch).gather(0, action_batch)

    # Compute loss

    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values)
    # Optimize the model

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)  # gradient clipping for numerical stability
    optimizer.step()


if __name__ == "__main__":

    env = RandomChimera(2, 2, include_spin=INCLUDE_SPIN)

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

            # Perform one step of the optimization (on the policy network)
            optimize_model()
            if done:  # it is done when model performs final spin flip
                break
        # Get n-step sum of rewards and and predicted rewards

        for t in range(len(trajectory)):
            # state, action reward_n, state_n
            stop = t + N if len(trajectory) - t > N else len(trajectory)-1

            reward_n = 0
            for k in range(t, stop + 1):
                reward_n += trajectory[k][2]
            memory.push(trajectory[t][0].to("cpu"), trajectory[t][1], reward_n, trajectory[stop][0].to("cpu"))
        steps_done += 1

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            validation_score_new = validate(validation_score, val_instance, q_values_global)
            print(validation_score_new)
            print(steps_done)

            if validation_score_new <= validation_score:
                validation_score = validation_score_new
                print(validation_score)
                torch.save({
                    'episode': episode,
                    'model_state_dict': policy_net.state_dict()}, PATH)

        torch.save({
            'episode': episode,
            'val_instance': val_instance,
            'val_score' : validation_score,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, CHECKPOINT_PATH)

