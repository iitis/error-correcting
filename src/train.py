
import networkx as nx
import torch
import math
import csv
import pickle
import random as rn
import torch.optim as optim
import torch.nn as nn

from src.environment import RandomChimera, ComputeChimera
from src.utils import nx_to_pytorch, TransitionMemory, n_step_transition, random_spin_flips
from src.DIRAC import DIRAC
from src.data_gen import generate_solved_chimera_from_csv
from itertools import count
from tqdm import tqdm
from torch_geometric.data import Batch
from torch_geometric.nn import DataParallel
from statistics import mean
from math import inf
from collections import deque

# Cuda devices
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
world_size = torch.cuda.device_count()
print('Let\'s use', world_size, 'GPUs!')

# Global constants
PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/models/model_C2_C4_v3.pt"
CHECKPOINT_PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/models/model_checkpoint.pt"
VAL_PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/chimera_512.pkl"
DATA_PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/random_C2_trajectory1.pkl"
BEST_VALUES_PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/chimera_512_checkpoint.pkl"

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 1750
NUM_EPISODES = 10000
TARGET_UPDATE = 10
N = 30
CHECKPOINT = True
INCLUDE_SPIN = True

# Models and optimizer
policy_net = DIRAC(include_spin=INCLUDE_SPIN)
target_net = DIRAC(include_spin=INCLUDE_SPIN)
#policy_net = DataParallel(policy_net)
#target_net = DataParallel(target_net)
policy_net = policy_net.to(device)
target_net = target_net.to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())

if CHECKPOINT:
    print("Loading checkpoint")
    checkpoint = torch.load(CHECKPOINT_PATH)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    target_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    episode_checkpoint = checkpoint["episode"]

    with open(BEST_VALUES_PATH, 'rb') as f:
        val_dataset = pickle.load(f)

# Global variables
env = RandomChimera((2, 2), include_spin=INCLUDE_SPIN)
steps_done = 0
validation_score = inf
memory = TransitionMemory(80000)  # n-step transition
q_values_global = None
improved = False
percentage = 0.05
best_energy = inf

if not CHECKPOINT:
    with open(VAL_PATH, 'rb') as f:
        val_dataset = pickle.load(f)

    episode_checkpoint = -1

# maybe use n-step transition?

# For start we wil train for 1000 episodes on C_2.
# Next step will be variable C (probably C_1 - C_5)


def select_action_epsilon_greedy(environment):

    global steps_done

    sample = rn.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)  # epsilon decay

    if sample > eps_threshold:
        return select_action_policy(environment)
    else:
        return rn.choice(environment.available_actions)


def select_action_policy(environment):

    global q_values_global
    state = environment.state.to(device)

    with torch.no_grad():
        # Here choice really depends on INCLUDE_SPIN, if True then q_values_global is None
        if q_values_global is None:
            q_values = policy_net(state)

            mask = torch.tensor(environment.mask, device=device)  # used to mask available actions
            q_values = torch.add(q_values, 1E-8)  # to avoid 0 * -inf
            action = mask * q_values
            return action.argmax().item()
        else:
            mask = torch.tensor(environment.mask, device=device)  # used to mask available actions
            q_values_global = torch.add(q_values_global, 1E-8)  # to avoid 0 * -inf
            action = mask * q_values_global
            return action.argmax().item()


def validate():
    global q_values_global
    global val_dataset

    choice = rn.randint(1, 10)
    improve = False
    val_set = val_dataset["{}".format(choice)]
    val_env = ComputeChimera(val_set, include_spin=INCLUDE_SPIN)
    min_eng = inf  # SERIOUS BUG

    if INCLUDE_SPIN:
        q_values_global = None
    else:
        with torch.no_grad():
            q_values_global = policy_net(val_env.state.to(device))

    for t in count():
        # Select and perform an action
        action = select_action_policy(val_env)
        _, _, done, _ = val_env.step(action)
        energy = val_env.energy()
        if energy < min_eng:
            min_eng = energy
            # update best instance
            spins = nx.get_node_attributes(val_env.chimera, "spin")
            nx.set_node_attributes(val_set, spins, "spin")
            val_dataset["{}".format(choice)] = val_set
            improve = True
        if done:  # it is done when model performs final spin flip
            break

    return improve


def validate_solved():
    global q_values_global
    global improved
    global percentage
    global best_energy

    sol_path = '/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/chimera_512_sol.pkl'
    chimera_path = '/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/chimera_512_002.csv'

    with open(sol_path, 'rb') as f:
        solution_dict = pickle.load(f)
    graph_solved = generate_solved_chimera_from_csv(chimera_path, solution_dict, 2)
    graph_fliped = random_spin_flips(graph_solved, percentage)

    val_env = ComputeChimera(graph_fliped, include_spin=INCLUDE_SPIN)
    min_eng = val_env.energy()
    old_best_energy = best_energy

    if INCLUDE_SPIN:
        q_values_global = None
    else:
        with torch.no_grad():
            q_values_global = policy_net(val_env.state.to(device))

    for t in count():
        # Select and perform an action
        action = select_action_policy(val_env)
        _, _, done, _ = val_env.step(action)
        energy = val_env.energy()
        if energy < min_eng:
            min_eng = energy
            # update best instance
            best_energy = min_eng if min_eng < best_energy else best_energy
            improved = True
        if done:  # it is done when model performs final spin flip
            break

    if improved:
        percentage = percentage - 0.01 if percentage > 0.01 else 0.01
        if percentage == 0.01:
            improved = True if best_energy < old_best_energy else False

    return improved

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = n_step_transition(*zip(*transitions))

    action_batch = torch.tensor(batch.action, device=device)
    reward_batch = torch.tensor(batch.reward_n, device=device)
    state_batch = Batch.from_data_list(batch.state).to(device)
    #stop_states = Batch.from_data_list(batch.state_n)
    expected_state_action_values = torch.tensor(batch.expected, device=device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(0, action_batch)

    # Compute Huber loss
    criterion = nn.MSELoss()
    loss = criterion(reward_batch + expected_state_action_values, state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)  # gradient clipping for numerical stability
    optimizer.step()


if __name__ == "__main__":

    for episode in tqdm(range(NUM_EPISODES), leave=None, desc="episodes"):

        if episode < episode_checkpoint:
            continue

        # Initialize the environment and state
        env.reset()
        trajectory = deque([], maxlen=10000)
        validation = False

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
            action = select_action_epsilon_greedy(env)
            _, reward, done, _ = env.step(action)

            # Store the transition in memory
            trajectory.append([state, action, reward])

            # Perform one step of the optimization (on the policy network)
            optimize_model()
            if done:  # it is done when model performs final spin flip
                break

        # Get n-step sum of rewards and and predicted rewards

        for k in range(len(trajectory)):
            # state, action reward_n, state_n
            stop = k + N if len(trajectory) - k > N else len(trajectory)-1
            expected = target_net(trajectory[stop][0].to(device)).max() * GAMMA

            reward_n = 0
            for i in range(k, stop + 1):
                reward_n = trajectory[i][2] + GAMMA * reward_n
            memory.push(trajectory[k][0], trajectory[k][1], reward_n, trajectory[stop][0], expected)

        steps_done += 1

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            validation = validate_solved()

            if validation:
                torch.save({
                    'episode': episode,
                    'model_state_dict': policy_net.state_dict()}, PATH)

        torch.save({
            'episode': episode,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, CHECKPOINT_PATH)

        #with open(BEST_VALUES_PATH, 'wb') as f:
        #    pickle.dump(val_dataset, f)
