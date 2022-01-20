import pickle
import torch
import math
import copy
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rn

from math import inf
from itertools import count
from tqdm import tqdm
from pathlib import Path
from src.environment import Chimera
from src.data_gen import generate_solved_chimera_from_csv, generate_chimera_from_txt, create_solution_dict
from src.DIRAC import DIRAC
from src.utils import compute_energy_nx, random_spin_flips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

MODEL_PATH = ROOT_DIR + "/models/model_C3.pt"
GROUND_PATH = ROOT_DIR + "/datasets/ground_states/"
PATH_128 = ROOT_DIR + "/datasets/128/"
PATH_128_GROUND = GROUND_PATH + "groundstates_128.txt"
PATH_512 = ROOT_DIR + "/datasets/512/"
PATH_512_GROUND = GROUND_PATH + "groundstates_512.txt"

model = DIRAC(include_spin=True).to(device)
model.load_state_dict(torch.load(MODEL_PATH)["model_state_dict"])
model.eval()
q_values_global = None

STEPS = 10

passes = 1

#create_solution_dict(PATH_512_GROUND, GROUND_PATH + "dict_512")

def select_action_policy(environment, q_values_global):

    state = environment.state.to(device)

    with torch.no_grad():
        # Here choice really depends on INCLUDE_SPIN, if True then q_values_global is None
        q_values = model(state) if q_values_global is None else q_values_global

        mask = torch.tensor(environment.mask, device=device)  # used to mask available actions
        q_values = torch.add(q_values, 1E-8)  # to avoid 0 * -inf
        action = mask * q_values
        return action.argmax().item()

def solve(nx_graph, mode, draw=False):
     env = Chimera(nx_graph, include_spin=True)
     graph = copy.deepcopy(env.chimera)
     min_eng = env.energy()
     energy_path = []
     energy_path.append(min_eng)
     for _ in count():
          # Select and perform an action
          action = rn.choice(env.available_actions) if mode == "rng" else select_action_policy(env, q_values_global)
          _, _, done, _ = env.step(action)
          energy = env.energy()
          energy_path.append(energy)
          if energy < min_eng:
            min_eng = energy
            graph = copy.deepcopy(env.chimera)
          if done:  # it is done when model performs final spin flip
               break
     return min_eng, graph

def eps_multi(nx_graph, t_max, eps):

    graph = copy.deepcopy(nx_graph)
    solution = compute_energy_nx(graph)
    for t in t_max:
        solution_new, graph_new = solve(graph, "model")
        if solution_new < solution:
            solution = solution_new
            graph = copy.deepcopy(graph_new)
        else:
            pass




with open(GROUND_PATH + "dict_512", 'rb') as f:
     solution_dict = pickle.load(f)

for mode in ["model"]:
    for percentage in [0.20, 0.15, 0.1, 0.05, 0.01]: #[0.1, 0.05, 0.01]:
        for p in range(passes):

            directory = os.fsencode(PATH_512)
            improved = []
            c = 0
            for file in tqdm(os.listdir(directory)):
                c +=1
                filename = os.fsdecode(file)
                number = int(filename[0:3])
                chimera = generate_solved_chimera_from_csv(PATH_512 + filename, solution_dict, number)
                chimera_disturbed = copy.deepcopy(random_spin_flips(chimera, percentage))
                start = compute_energy_nx(chimera_disturbed)
                solution = solve(chimera_disturbed, mode)
                if solution < start:
                    improved.append(1)
                else:
                    improved.append(0)
                if c >= 30:
                    break

            print("{} for diruption {} after {} passes: {}".format(mode, percentage, p, sum(improved)/30))


