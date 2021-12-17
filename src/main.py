"""
Extremely ugly now. I will make it pretty later
"""

import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from math import inf
import pandas as pd
from itertools import count
from src.data_gen import generate_chimera_from_csv
from src.DIRAC import DIRAC
from src.environment import ComputeChimera

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/models/model_C2_no_spin.pt"
CHECKPOINT_PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/models/model_checkpoint.pt"
VAL_PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/chimera_512.pkl"


def select_action_policy(environment):

    global q_values_global
    state = environment.state.to(device)

    with torch.no_grad():
        # Here choice really depends on INCLUDE_SPIN, if True then q_values_global is None
        if q_values_global is None:
            q_values = policy_net(state)

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
    model = policy_net = DIRAC(include_spin=False).to(device)
    model.load_state_dict(torch.load(PATH)["model_state_dict"])
    model.eval()

    with open(VAL_PATH, 'rb') as f:
        dataset = pickle.load(f)

    for choice in range(1, 11):  # expected time 2m30s for 10 chimeras_512

        energy_path = []
        val_set = dataset["{}".format(choice)]
        val_env = ComputeChimera(val_set, include_spin=False)
        min_eng = inf
        q_values_global = policy_net(val_env.state.to(device))

        for t in tqdm(count()):
            # Select and perform an action
            action = select_action_policy(val_env)
            _, _, done, _ = val_env.step(action)
            energy = val_env.energy()
            energy_path.append(energy)
            if energy < min_eng:
                min_eng = energy
            if done:  # it is done when model performs final spin flip
                break
        x = np.arange(val_env.chimera.number_of_nodes())
        min_eng = [min_eng for _ in range(val_env.chimera.number_of_nodes())]
        plt.plot(x, energy_path, x, min_eng)
        plt.show()

    """
    dataset = {}
    for i in range(1, 10):
        graph = generate_chimera_from_csv("/home/tsmierzchalski/pycharm_projects/error-correcting/datasets"
                                          "/chimera_512_00{}.csv".format(i))
        dataset["{}".format(i)] = graph
    dataset["10"] = generate_chimera_from_csv("/home/tsmierzchalski/pycharm_projects/error-correcting/datasets"
                                              "/chimera_512_010.csv")

    with open('/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/chimera_512.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    """