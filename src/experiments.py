import pickle
import torch
import math
import copy
import os
import ast


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rn
import dwave_networkx as dnx
import networkx as nx

from math import inf
from itertools import count
from tqdm import tqdm
from pathlib import Path
from src.environment import Chimera
from src.data_gen import generate_solved_chimera_from_csv, generate_chimera_from_txt, create_solution_dict
from src.DIRAC import DIRAC
from src.utils import compute_energy_nx, random_spin_flips, nx_to_pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

MODEL_PATH = ROOT_DIR + "/models/model_C3.pt"
GROUND_PATH = ROOT_DIR + "/datasets/ground_states/"
PATH_128 = ROOT_DIR + "/datasets/128/"
PATH_128_GROUND = GROUND_PATH + "groundstates_128.txt"
PATH_512 = ROOT_DIR + "/datasets/512/"
PATH_512_GROUND = GROUND_PATH + "groundstates_512.txt"
PATH_1152 = ROOT_DIR + "/datasets/1152/"
PATH_1152_GROUND = GROUND_PATH + "groundstates_1152.txt"
PATH_2048 = ROOT_DIR + "/datasets/2048/"
PATH_2048_GROUND = GROUND_PATH + "groundstates_2048.txt"
D_WAVE_PATH = ROOT_DIR + "/datasets/d_wave/"

model = DIRAC(include_spin=True).to(device)
model.load_state_dict(torch.load(MODEL_PATH)["model_state_dict"])
model.eval()
q_values_global = None

STEPS = 10

passes = 1

#create_solution_dict(PATH_2048_GROUND, GROUND_PATH + "dict_2048")

def select_action_policy(environment, q_values_global):

    state = environment.state.to(device)

    with torch.no_grad():
        # Here choice really depends on INCLUDE_SPIN, if True then q_values_global is None
        q_values = model(state) if q_values_global is None else q_values_global

        mask = torch.tensor(environment.mask, device=device)  # used to mask available actions
        q_values = torch.add(q_values, 1E-8)  # to avoid 0 * -inf
        action = mask * q_values
        return action.argmax().item()

def solve(nx_graph, draw=False):
    env = Chimera(nx_graph, include_spin=True)
    graph = copy.deepcopy(env.chimera)
    min_eng = env.energy()
    energy_path = []
    energy_path.append(min_eng)
    for _ in count():
        # Select and perform an action
        action = select_action_policy(env, q_values_global) #rn.choice(env.available_actions)
        _, _, done, _ = env.step(action)
        energy = env.energy()
        energy_path.append(energy)
        if energy < min_eng:
            min_eng = energy
            graph = copy.deepcopy(env.chimera)
        if done:  # it is done when model performs final spin flip
            break
    x = np.arange(env.action_space.n + 1)
    y = [min_eng for _ in x]
    plt.plot(x, energy_path, x, y)
    plt.show()

    return min_eng, graph



def sim_dirac(nx_graph, iter_max, temp, temp_cut):
    env = Chimera(nx_graph, True)
    best = copy.deepcopy(env.chimera)
    for i in range(iter_max):

        t = temp / float(i + 1)
        if t > temp_cut:
            action = rn.randint(0, prop.number_of_nodes() - 1)
            prop.nodes[action]["spin"] *= -1
            diff = compute_energy_nx(prop) - compute_energy_nx(curr)
            if compute_energy_nx(prop) < compute_energy_nx(curr):
                curr = copy.deepcopy(prop)
                if compute_energy_nx(curr) < compute_energy_nx(best):
                    best = copy.deepcopy(curr)

            elif math.exp(-diff / t) > rn.random():
                curr = copy.deepcopy(prop)
        else:
            state = nx_to_pytorch(curr, True)
            action = model(state).argmax().item()
            curr.nodes[action]["spin"] *= -1

def sim(nx_graph, iter_max, temp):

    curr = copy.deepcopy(nx_graph)
    best = copy.deepcopy(curr)
    for i in range(iter_max):
        prop = copy.deepcopy(curr)
        t = temp / float(i + 1)
        action = rn.randint(0, prop.number_of_nodes()-1)
        prop.nodes[action]["spin"] *= -1
        diff = compute_energy_nx(prop) - compute_energy_nx(curr)
        if compute_energy_nx(prop) < compute_energy_nx(curr):
            curr = copy.deepcopy(prop)
            if compute_energy_nx(curr) < compute_energy_nx(best):
                best = copy.deepcopy(curr)

        elif math.exp(-diff/t) > rn.random():
            curr = copy.deepcopy(prop)
    return compute_energy_nx(best)

df = pd.read_csv(D_WAVE_PATH + "512_1.csv", index_col=0)

instance = 1

spins = df["sample"][instance]
spins = ast.literal_eval(spins)


external = df["h"][9]
external = ast.literal_eval(external)

edge_attr = df["J"][9]
edge_attr = ast.literal_eval(edge_attr)

energy = df["energy"][instance]

chimera = dnx.chimera_graph(8,8,4)

nx.set_node_attributes(chimera, external, "external")
nx.set_node_attributes(chimera, spins, "spin")
nx.set_edge_attributes(chimera, edge_attr, "coupling")

sol = sim_dirac(chimera, 1000, 10, 0.1)
sol2 = sim(chimera, 1000, 10)

print(energy)
print(sol)
print(sol2)

#with open(GROUND_PATH + "dict_2048", 'rb') as f:
#    solution_dict = pickle.load(f)

#chimera = generate_solved_chimera_from_csv(PATH_2048 + "002.txt", solution_dict, 2)


#solution, grAPH = solve(chimera)

#print(solution)
#print(energy)

"""
with open(GROUND_PATH + "dict_2048", 'rb') as f:
     solution_dict = pickle.load(f)

for mode in ["model"]:
    for percentage in [0.1, 0.05, 0.01]: #[0.1, 0.05, 0.01]:
        for p in range(passes):

            directory = os.fsencode(PATH_2048)
            improved = []
            c = 0
            for file in tqdm(os.listdir(directory)):
                c +=1
                filename = os.fsdecode(file)
                number = int(filename[0:3])
                chimera = generate_solved_chimera_from_csv(PATH_2048 + filename, solution_dict, number)
                chimera_disturbed = copy.deepcopy(random_spin_flips(chimera, percentage))
                start = compute_energy_nx(chimera_disturbed)
                solution, graph = solve(chimera_disturbed, mode)
                if solution < start:
                    improved.append(1)
                else:
                    improved.append(0)
                if c >= 20:
                    break

            print("{} for diruption {} after {} passes: {}".format(mode, percentage, p, sum(improved)/20))
"""

