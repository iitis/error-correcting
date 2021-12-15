import torch
import math
import csv
import random as rn
import torch.optim as optim
import torch.nn as nn
from src.environment import RandomChimera
from src.utils import nx_to_pytorch
from src.DIRAC import DIRAC
from itertools import count
from tqdm import tqdm
from torch_geometric.data import Batch
from statistics import mean

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# global constants
PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/model_big.pt"
CHECKPOINT_PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/model_checkpoint.pt"
VAL_PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/graph_reward_track.pt"

BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 225
NUM_EPISODES = 1000
TARGET_UPDATE = 10
CHECKPOINT = False

# Models and optimizer
policy_net = DIRAC().to(device)
target_net = DIRAC().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters())

if CHECKPOINT:
    checkpoint = torch.load(PATH)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    target_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Global variables
env = RandomChimera((2, 2))
steps_done = 0

# maybe use n-step transition?

# For start we wil train for 1000 episodes on C_2.
# Next step will be variable C (probably C_1 - C_5)


def chose_action(environment):

    global steps_done

    sample = rn.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)  # epsilon decay
    if sample > eps_threshold:
        with torch.no_grad():
            pass
    else:
        return rn.choice(environment.available_actions)


if __name__ == "main":

    for episode in range(NUM_EPISODES):

        # Initialize the environment and state
        env.reset()






"""

steps_done = 0

csv_columns = ["episode", "step", "energy"]
dict_data = []
csv_file = "/home/tsmierzchalski/pycharm_projects/error-correcting/rewards.csv"


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions)) # TO DO better batches in replay memory.

    action_batch = torch.tensor(batch.action, device=device)
    reward_batch = torch.tensor(batch.reward, device=device)
    state_batch = Batch.from_data_list(batch.state)
    next_states = Batch.from_data_list(batch.next_state)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(0, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net

    next_state_values = target_net(next_states).detach()  # detach from computation graph so we dont compute gradient
    next_state_values = torch.reshape(next_state_values, (BATCH_SIZE, -1))
    next_state_values = torch.max(next_state_values, 1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)  # gradient clipping for numerical stability
    optimizer.step()


def select_action(state, env):  # write it better, env shouldn't be necessary
    global steps_done

    sample = rn.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)  # epsilon decay
    if sample > eps_threshold:
        with torch.no_grad():
            mask = torch.tensor(env.actions_taken, device=device)  # used to mask available actions
            q_values = policy_net(state)
            q_values = torch.add(q_values, 0.001)  # to avoid 0 * -inf
            action = mask * q_values
            return action.argmax().item()
    else:
        return rn.choice(env.available_actions)


def solve(data):

    env_local = IsingGraph2d(data)
    energy = math.inf
    state = env_local.data
    for t in count():
        # Select and perform an action
        action = select_action(state, env_local)
        next_state, _, done, _ = env_local.step(action)

        # compute energy
        new_energy = compute_energy(state)
        if new_energy < energy:
            energy = new_energy

        # Move to the next state
        state = next_state

        if done:  # it is done when model performs final spin flip
            break

    return energy


def validate():
    global available_actions
    val_set = torch.load(VAL_PATH)
    val_list = []
    for i in range(10):
        graph = val_set["{}".format(i)]
        energy = solve(graph)
        val_list.append(energy)

    return mean(val_list)


for episode in tqdm(range(NUM_EPISODES)):
    available_actions = list(range(env.action_space.n))  # reset
    # Initialize the environment and state
    env.reset()
    state = env.data
    validation_score = math.inf
    for t in count():
        # Select and perform an action
        action = select_action(state, env)
        next_state, reward, done, action_taken = env.step(action)

        
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:  # it is done when model performs final spin flip
            break

    steps_done += 1

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        validation_score_new = validate()

        if validation_score_new < validation_score:
            validation_score = validation_score_new
            torch.save({
                'episode': episode,
                'validation_score': validation_score,
                'model_state_dict': policy_net.state_dict()}, PATH)

    torch.save({
        'episode': episode,
        'model_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, CHECKPOINT_PATH)

"""


