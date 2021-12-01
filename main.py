"""
Extremely ugly now. I will make it pretty later
"""

import torch
import math
import csv
from tqdm import tqdm
from learn import DIRAC
from data_gen import generate_ising_lattice
from enviroment import IsingGraph2dRandom, IsingGraph2d
from utils import plot_graph, gauge_transformation, compute_energy
import pandas as pd
#import h5py
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/model.pt"
SAVE = "/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/graph_reward_track.pt"

policy_net = DIRAC().to(device)
checkpoint = torch.load(PATH)
policy_net.load_state_dict(checkpoint['model_state_dict'])

csv_file = "/home/tsmierzchalski/pycharm_projects/error-correcting/energy_trained_graphs.csv"

df = pd.DataFrame()


def select_action_trained(env, state):

    global available_actions
    mask = torch.tensor(env.actions_taken, device=device)
    q_values = policy_net(state)
    q_values = torch.add(q_values, 0.001)  # to avoid 0 * -inf
    action = mask * q_values
    return action.argmax().item()


if __name__ == '__main__':
    env = IsingGraph2dRandom((10, 10))
    data = {}
    for i in tqdm(range(10)):
        available_actions = list(range(env.action_space.n))  # reset
        energy_list = []
        # Initialize the environment and state
        env.reset()
        state = env.data
        energy = math.inf
        data["{}".format(i)] = state
        for t in tqdm(range(env.action_space.n)):
            # Select and perform an action

            action = select_action_trained(env, state)
            next_state, reward, done, action_taken = env.step(action)

            # Remove action taken from available actions

            available_actions.remove(action_taken)

            # add data to csv
            new_energy = compute_energy(state)
            energy_list.append(new_energy)

            if new_energy < energy:
                energy = new_energy

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)

            if done:  # it is done when model performs final spin flip
                break
        df["{}".format(i)] = energy_list
    df.to_csv(csv_file, sep=',',)
    torch.save(data, SAVE)

    """
        try:
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for data in dict_data:
                    writer.writerow(data)
        except IOError:
            print("I/O error")
    """

            if done:  # it is done when model performs final spin flip
                break
        df["{}".format(i)] = energy_list
    df.to_csv(csv_file, sep=',',)
    torch.save(data, SAVE)

    """
        try:
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for data in dict_data:
                    writer.writerow(data)
        except IOError:
            print("I/O error")
    """



