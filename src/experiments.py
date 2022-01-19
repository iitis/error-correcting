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

model = DIRAC(include_spin=True).to(device)
model.load_state_dict(torch.load(MODEL_PATH)["model_state_dict"])
model.eval()
q_values_global = None

STEPS = 10

#create_solution_dict(PATH_128_GROUND, GROUND_PATH + "dict_128")

def select_action_policy(environment, q_values_global):

    state = environment.state.to(device)

    with torch.no_grad():
        # Here choice really depends on INCLUDE_SPIN, if True then q_values_global is None
        q_values = model(state) if q_values_global is None else q_values_global

        mask = torch.tensor(environment.mask, device=device)  # used to mask available actions
        q_values = torch.add(q_values, 1E-8)  # to avoid 0 * -inf
        action = mask * q_values
        return action.argmax().item()

def solve(nx_graph):
     env = Chimera(nx_graph, include_spin=True)
     min_eng = math.inf
     energy_path = []
     energy = env.energy()
     energy_path.append(energy)
     for _ in count():
          # Select and perform an action
          action = rn.choice(env.available_actions)#select_action_policy(env, q_values_global)
          _, _, done, _ = env.step(action)
          energy = env.energy()
          energy_path.append(energy)
          if energy < min_eng:
               min_eng = energy
          if done:  # it is done when model performs final spin flip
               break
     return min_eng, energy_path

with open(GROUND_PATH + "dict_128", 'rb') as f:
     solution_dict = pickle.load(f)

directory = os.fsencode(PATH_128)
improved = []
ground_state = []
failed = []
for file in tqdm(os.listdir(directory)):
    filename = os.fsdecode(file)
    number = int(filename[0:3])
    chimera = generate_solved_chimera_from_csv(PATH_128 + filename, solution_dict, number)
    chimera_disturbed = copy.deepcopy(random_spin_flips(chimera, 0.1))
    start = compute_energy_nx(chimera_disturbed)
    solution, energy_path = solve(chimera_disturbed)
    ground = solution_dict[str(number)]["ground"]
    if solution < start:
        improved.append(1)
    else:
        improved.append(0)
        failed.append(number)


    if  math.isclose(solution, ground, abs_tol=10**-3):
         ground_state.append(1)
    else:
        ground_state.append(0)

print(sum(improved)/100, sum(ground_state)/100)
print(failed)

