"""
Simple script for generating plots etc.
"""

import torch
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd


def cor(x):
    math.isclose(chimera2048_no_errors['energies'], energies["value"], abs_tol=10 ** -4)


chimera2048_path = "/home/tsmierzchalski/pycharm_projects/error-correcting/chimera2048.csv"
ground_path = "/home/tsmierzchalski/pycharm_projects/error-correcting/groundstates_otn2d.csv"
no_errors_path = "/home/tsmierzchalski/pycharm_projects/error-correcting/chimera2048_no_errors.csv"

chimera2048_no_errors = pd.read_csv(no_errors_path, delimiter=",", index_col=0)
chimera2048_no_errors["time"] = pd.to_numeric(chimera2048_no_errors["time"])

ground = pd.read_csv(ground_path, delimiter=" ", header=None)
ground = ground[2]



energies = ground.iloc[chimera2048_no_errors["i"]-1]
energies = energies.reset_index()
energies = energies[2]
energies = energies.round(3)
energies = energies.to_frame(name="value")


chimera2048_no_errors = chimera2048_no_errors.reset_index()
chimera2048_no_errors["energies"] = chimera2048_no_errors["energies"].round(3)
chimera2048_no_errors['correct'] = np.where(chimera2048_no_errors['energies'] == energies["value"], True, False)

chimera2048_approx_correct = chimera2048_no_errors.loc[chimera2048_no_errors['correct'] == True]
chimera2048_approx_correct.to_csv("/home/tsmierzchalski/pycharm_projects/error-correcting/chimera2048_approx_correct.csv")



"""
chimera2048_no_errors['correct'] = np.where(chimera2048_no_errors['energies'] == energies["value"], True, False)
means = chimera2048_no_errors.groupby('Sparsity')['time'].mean()
chimera2048 = chimera2048.loc[chimera2048["time"] != "ERROR"]
ground = ground[2]
instance = [x for x in range(1,100)]
ground["instance"] = instance

dense = chimera2048.loc[chimera2048["Sparsity"] == "Dense"]
sparse = chimera2048.loc[chimera2048["Sparsity"] == "Sparse"]

dense = dense[["Sparsity", "time"]]
sparse = sparse[["Sparsity", "time"]]


#correct = chimera2048.loc[chimera2048["i"] == ground["instance"]]
"""