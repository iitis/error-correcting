import torch
import math
import csv
import random as rn
import torch.optim as optim
import torch.nn as nn
from enviroment import IsingGraph2dRandom
from utils import compute_energy
from learn import DIRAC, ReplayMemory, Transition
from itertools import count
from tqdm import tqdm
from torch_geometric.data import Batch

PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/model2.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 225
NUM_EPISODES = 1000
TARGET_UPDATE = 10

policy_net = DIRAC().to(device)
target_net = DIRAC().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters())

#checkpoint = torch.load(PATH)
#policy_net.load_state_dict(checkpoint['model_state_dict'])
#target_net.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


env = IsingGraph2dRandom((3, 3))

memory = ReplayMemory(10000)
available_actions = list(range(env.action_space.n))
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


def select_action(state):
    global steps_done
    global available_actions
    sample = rn.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)  # epsilon decay
    if sample > eps_threshold:
        with torch.no_grad():
            mask = torch.tensor(env.actions_taken, device=device)
            q_values = policy_net(state)
            q_values = torch.add(q_values, 0.001)  # to avoid 0 * -inf
            action = mask * q_values
            return action.argmax().item()
    else:
        return rn.choice(available_actions)

def validate():
    pass

for episode in tqdm(range(NUM_EPISODES)):
    available_actions = list(range(env.action_space.n))  # reset
    # Initialize the environment and state
    env.reset()
    state = env.data
    old_energy = math.inf
    for t in count():
        # Select and perform an action
        action = select_action(state)
        next_state, reward, done, action_taken = env.step(action)
        
        # Remove action taken from available actions

        available_actions.remove(action_taken)
        
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # add data to csv
        new_energy = compute_energy(state)
        if new_energy < old_energy:
            dict_data.append({"episode": episode, "step": steps_done, "energy": new_energy})
            old_energy = new_energy

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()  # TO DO: optimize_model()
        if done:  # it is done when model performs final spin flip
            break

    steps_done += 1

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    torch.save({
        'episode': episode,
        'model_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, PATH)


    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")
