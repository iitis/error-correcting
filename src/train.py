import torch
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


# Cuda devices
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
world_size = torch.cuda.device_count()
#print('Let\'s use', world_size, 'GPUs!')

# Global constants
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = ROOT_DIR + "/models/model_C3_v3.pt"
CHECKPOINT_PATH = ROOT_DIR + "/models/model_checkpoint.pt"
D_WAVE_PATH = ROOT_DIR + "/datasets/d_wave/"

BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.05
NUM_EPISODES = 100000
EPS_DECAY = int(NUM_EPISODES * 0.20)
TARGET_UPDATE = 10
N = 10
CHECKPOINT = False
INCLUDE_SPIN = True

# Models and optimizer
policy_net = DIRAC(include_spin=INCLUDE_SPIN)
target_net = DIRAC(include_spin=INCLUDE_SPIN)
target_net.load_state_dict(policy_net.state_dict())
#policy_net = DataParallel(policy_net)
#target_net = DataParallel(target_net)
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
    validation_score = checkpoint['val_score']

else:
    episode_checkpoint = -1
    validation_score = inf

# Global variables
steps_done = 0

memory = TransitionMemory(300000)  # n-step transition, will have aprox. 70 GB size
q_values_global = None

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

        if episode % 10 == 0:
            x =  np.arange((episode/TARGET_UPDATE) + 1)
            print(x)
            y1 = val_q_list
            y2 = val_hard_q_list
            print(y1, y2)
            plt.plot(x,y1,x,y2)
            plt.show()

        if sum_loss < validation_score:
            validation_score = sum_loss
            print(validation_score)
            torch.save({
                'episode': episode,
                'model_state_dict': policy_net.state_dict()}, MODEL_PATH)



        torch.save({
            'episode': episode,
            'val_set': val_set,
            'val_set_hard': val_set_hard,
            'val_score' : validation_score,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, CHECKPOINT_PATH)

