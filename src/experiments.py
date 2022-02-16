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
from src.data_gen import generate_chimera_from_csv_dwave, generate_chimera
from src.DIRAC import DIRAC
from src.utils import compute_energy_nx, random_spin_flips, nx_to_pytorch
from src.train import DQNTrainer

root_dir = Path.cwd().parent
model_path = root_dir / "models" / "model_C3_v5.pt"
model_validation_path = root_dir / "models" / "model_C3_v4_val.pt"
checkpoint_path = root_dir / "models" / "model_checkpoint_test.pt"
dwave_path = root_dir / "datasets" / "d_wave"

def solve(chimera: nx.Graph) -> float:

    env = Chimera(chimera, include_spin=True)
    model = DQNTrainer(env)
    model.load_model(model_path)
    min_energy: float = env.energy()
    energy_list: list = [min_energy]
    improving = True
    while improving:
        improving = False
        for _ in count():
            # Select and perform an action
            action = model.select_action_on_policy()
            _, _, done, _ = model.env.step(action)
            energy = model.env.energy()
            energy_list.append(energy)
            if energy < min_energy:
                min_energy = energy
                improving = True
            # it is done when model performs final spin flip
            if done:
                break

    return min_energy
env = Chimera(generate_chimera(3,3), include_spin=True)
model = DQNTrainer(env)
model.load_model(checkpoint_path)
model.set_val_env(3,3)
print(model.val_env.simulated_annealing(100,10))
model.plot_energy_path()