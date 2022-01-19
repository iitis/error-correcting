import pickle
import torch
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
from src.data_gen import generate_solved_chimera_from_csv, generate_chimera_from_txt
from src.DIRAC import DIRAC
from src.utils import compute_energy_nx, random_spin_flips

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

MODEL_PATH = ROOT_DIR + "/models/model_C3.pt"
GROUND_PATH = ROOT_DIR + "/datasets/ground_states/"
PATH_128 = ROOT_DIR + "/datasets/128/"

model = DIRAC(include_spin=True).to(device)
model.load_state_dict(torch.load(MODEL_PATH)["model_state_dict"])
model.eval()
q_values_global = None

STEPS = 10

directory = os.fsencode(PATH_128)

#chimera = pd.read_csv(PATH_128 + "001.txt", skiprows=[0], sep=" ", header=None)


for file in os.listdir(directory):
     filename = os.fsdecode(file)
     chimera = generate_chimera_from_txt(PATH_128 + filename)
     print(chimera)