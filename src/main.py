"""
Extremely ugly now. I will make it pretty later
"""

import torch
import math
from tqdm import tqdm
from learn import DIRAC
from enviroment import IsingGraph2dRandom, IsingGraph2d
from utils import compute_energy
import pandas as pd
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/model_big.pt"
VAL_PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/graph_reward_track.pt"
SAVE = "/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/random_15_100"

policy_net = DIRAC().to(device)
checkpoint = torch.load(PATH)
policy_net.load_state_dict(checkpoint['model_state_dict'])

csv_file = "/home/tsmierzchalski/pycharm_projects/error-correcting/test_15_100.csv"

df = pd.DataFrame(columns=['lowest_energy', 'e0', 'energy_path'])
graph_dict = {}



def select_action_trained(state, env):

    mask = torch.tensor(env.actions_taken, device=device)
    q_values = policy_net(state)
    q_values = torch.add(q_values, 0.001)  # to avoid 0 * -inf
    action = mask * q_values
    return action.argmax().item()


def solve_trained(data):

    env_local = IsingGraph2d(data)
    energy_path = []
    lowest_energy = math.inf
    state = env_local.data
    for t in count():
        # Select and perform an action
        action = select_action_trained(state, env_local)
        next_state, _, done, _ = env_local.step(action)

        # compute energy
        new_energy = compute_energy(state)
        energy_path.append(new_energy)
        if new_energy < lowest_energy:
            lowest_energy = new_energy

        # Move to the next state
        state = next_state

        if done:  # it is done when model performs final spin flip
            break

    return lowest_energy, energy_path


if __name__ == '__main__':
    env = IsingGraph2dRandom((15, 15))
    for i in tqdm(range(100)):

        # Initialize the environment and state
        env.reset()
        starting_state = env.data
        graph_dict["{}".format(i)] = env.data
        #compute energy and energy_path
        lowest_energy, energy_path = solve_trained(starting_state)
        e0 = lowest_energy/env.data.num_nodes
        df = df.append({'lowest_energy': lowest_energy, "e0": e0, "energy_path": energy_path},
                       ignore_index=True)

        #it will be overwritten for every iteration
        df.to_csv(csv_file)
        torch.save(graph_dict, SAVE)


    """
    for i in tqdm(range(20)):

        # Initialize the environment and state
        env.reset()
        starting_state = env.data
        energy = math.inf

        new_energy = solve_trained(starting_state)
        

        energy_list = []
        # Initialize the environment and state
        env.reset()
        state = env.data
        energy = math.inf
        data["{}".format(i)] = state
        for t in count():
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