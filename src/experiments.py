import pickle
import torch
import copy

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rn

from math import inf
from itertools import count
from tqdm import tqdm
from src.environment import Chimera
from src.data_gen import generate_solved_chimera_from_csv, generate_chimera_from_csv
from src.DIRAC import DIRAC
from src.utils import compute_energy_nx, random_spin_flips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STEPS = 6

model_path = "/home/tsmierzchalski/pycharm_projects/error-correcting/models/model_C2_C4_v3.pt"
sol_path = '/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/chimera_512_sol.pkl'
chimera_path = '/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/chimera_512_002.csv'

with open(sol_path, 'rb') as f:
    solution_dict = pickle.load(f)

model = DIRAC(include_spin=True).to(device)
model.load_state_dict(torch.load(model_path)["model_state_dict"])
model.eval()
q_values_global = None

chimera_csv = pd.read_csv(chimera_path, header=None, index_col=0)

graph_random = generate_chimera_from_csv(chimera_path)
graph_solved = generate_solved_chimera_from_csv(chimera_path, solution_dict, 2)

graph_fliped = random_spin_flips(graph_solved, 0.10)



def select_action_policy(environment):

    global q_values_global
    state = environment.state.to(device)

    with torch.no_grad():
        # Here choice really depends on INCLUDE_SPIN, if True then q_values_global is None
        if q_values_global is None:
            q_values = model(state)
            mask = torch.tensor(environment.mask, device=device)  # used to mask available actions
            q_values = torch.add(q_values, 1E-8)  # to avoid 0 * -inf
            action = mask * q_values
            return action.argmax().item()
        else:
            mask = torch.tensor(environment.mask, device=device)  # used to mask available actions
            q_values_global = torch.add(q_values_global, 1E-8)  # to avoid 0 * -inf
            action = mask * q_values_global
            return action.argmax().item()



if __name__ == "__main__":
    for type in ["model"]:
        graph = copy.deepcopy(graph_fliped)
        q_values_global = None  # policy_net(val_env.state.to(device))
        min_eng = inf
        min_eng_plot = []
        min_eng_plot_global = []
        for i in range(STEPS):
            val_env = Chimera(graph, include_spin=True)
            # val_env.reset()
            old_min_energy = min_eng
            energy_path = []
            for t in tqdm(count()):
                # Select and perform an action
                action = select_action_policy(val_env) if type == "model" else rn.choice(val_env.available_actions) #
                _, _, done, _ = val_env.step(action)
                energy = val_env.energy()
                energy_path.append(energy)
                if energy < min_eng:
                    min_eng = energy
                    graph = copy.deepcopy(val_env.chimera)
                if done:  # it is done when model performs final spin flip
                    break
            min_eng_plot_global.append(min_eng)

            x = np.arange(val_env.chimera.number_of_nodes())
            min_eng_plot = [min_eng for _ in range(val_env.chimera.number_of_nodes())]
            plt.plot(x, energy_path, x, min_eng_plot)
            plt.xlabel("steps")
            plt.ylabel("energy")
            plt.title('instance 1, try {}, {}'.format(i + 1, type))
            plt.show()
            print(compute_energy_nx(graph))
            del val_env


        x = np.arange(STEPS)
        plt.plot(x, min_eng_plot_global)
        plt.xlabel("steps")
        plt.ylabel("energy")
        plt.title('instance 1 after {} repetitions, {}'.format(STEPS, type))
        plt.show()



