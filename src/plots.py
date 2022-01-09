"""
Simple script for generating plots etc.
"""

import torch
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import itertools

from src.data_gen import generate_chimera_from_csv

PATH = "/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/chimera_512_001.txt"

chimera = pd.read_csv(PATH)
print(chimera)






"""
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 3500
NUM_EPISODES = 20000

def func(x):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * x / EPS_DECAY)

x = np.arange(NUM_EPISODES)

plt.plot(x, func(x))
plt.show()
"""

"""
chimera_2048_1_path = "/home/tsmierzchalski/pycharm_projects/error-correcting/csv/chimera2048.csv"
chimera_2048_2_path = "/home/tsmierzchalski/pycharm_projects/error-correcting/csv/chimera2048-2.csv"
ground_path = "/home/tsmierzchalski/pycharm_projects/error-correcting/csv/groundstates_otn2d.txt"

chimera_2048_1 = pd.read_csv(chimera_2048_1_path, delimiter=";")
chimera_2048_2 = pd.read_csv(chimera_2048_2_path, delimiter=";")

def cor(x):
    return x+1

chimera_2048_2["i"] = chimera_2048_2["i"].apply(cor)

chimera_2048 = pd.concat([chimera_2048_1, chimera_2048_2], ignore_index=True)


chimera_2048["energies"] = chimera_2048["energies"].round(3)
ground = pd.read_csv(ground_path, delimiter=" ", header=None)
ground = ground[2]
energies = ground.iloc[chimera_2048["i"]-1]
energies = energies.reset_index()
energies = energies[2]
energies = energies.round(3)
energies = energies.to_frame(name="value")

chimera_2048['correct'] = np.where(chimera_2048['energies'] == energies["value"], True, False)

print(chimera_2048.loc[chimera_2048["correct"]==True])
bond = chimera_2048.groupby("bd")

for x in [16,24,32,64]:


    bond16 = bond.get_group(x)

    bond16_str = bond16.groupby("Strategy")
    for y in ["SVDTruncate", "MPSAnnealing"]:
        bond16_str_SVD = bond16_str.get_group(y)

        bond16_str_SVD_gauges = bond16_str_SVD.groupby("Layout")

        bond16_str_SVD_eg = bond16_str_SVD_gauges.get_group("EnergyGauges")
        bond16_str_SVD_ge = bond16_str_SVD_gauges.get_group("GaugesEnergy")
        bond16_str_SVD_ege = bond16_str_SVD_gauges.get_group("EngGaugesEng")

        bond16_str_SVD_eg.to_csv("/home/tsmierzchalski/pycharm_projects/error-correcting/csv/2048bond{}_{}_eg.csv".format(x, y))
        bond16_str_SVD_ge.to_csv("/home/tsmierzchalski/pycharm_projects/error-correcting/csv/2048bond{}_{}_A_ge.csv".format(x, y))
        bond16_str_SVD_ege.to_csv("/home/tsmierzchalski/pycharm_projects/error-correcting/csv/2048bond{}_{}_A_ege.csv".format(x, y))




"""
"""
chimera2048_no_errors = pd.read_csv(no_errors_path, delimiter=",", index_col=0)
chimera_correct = pd.read_csv(correct_path, delimiter=",", index_col=0)
gauges_energy = chimera_correct.loc[chimera2048_no_errors["Layout"] == "GaugesEnergy"]
energy_gauges = chimera_correct.loc[chimera2048_no_errors["Layout"] == "EnergyGauges"]
eng_gauges_eng = chimera_correct.loc[chimera2048_no_errors["Layout"] == "EngGaugesEng"]

betas_ge = gauges_energy.groupby('β').size()
betas_eg = energy_gauges.groupby('β').size()
betas_ege = eng_gauges_eng.groupby('β').size()
print(betas_ge)
print(betas_eg)
print(betas_ege)

print([x/176 for x in betas_ege])

x = list(range(1, 9))  # Sample data.
y1 = [0, 0.022727272727272728, 0.045454545454545456, 0.022727272727272728, 0.011363636363636364, 0.022727272727272728, 0, 0]
y2 = [0, 0.022727272727272728, 0.056818181818181816, 0.022727272727272728, 0.022727272727272728, 0.011363636363636364, 0.011363636363636364, 0.005681818181818182]
y3 = [0, 0.022727272727272728, 0.09090909090909091, 0.011363636363636364, 0, 0.03409090909090909, 0.028409090909090908, 0.011363636363636364]
plt.plot(x, y1, label='GaugesEnergy')
plt.plot(x, y2, label='EnergyGauges')
plt.plot(x, y3, label='EngGaugesEng')
plt.xlabel('β')
plt.ylabel('Probability (as fraction)')
plt.title("Probability of finding ground state")
plt.legend()
plt.savefig('/home/tsmierzchalski/pycharm_projects/error-correcting/probability.png')
"""
"""

  # Plot some data on the (implicit) axes.
plt.plot(x, x**2, label='quadratic')  # etc.
plt.plot(x, x**3, label='cubic')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Simple Plot")
plt.legend()


"""
"""
errors = chimera2048.loc[chimera2048["energies"] == "ERROR"]
errors.to_csv("/home/tsmierzchalski/pycharm_projects/error-correcting/errors.csv")

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


chimera2048_no_errors = chimera2048.loc[chimera2048["energies"] != "ERROR"]
chimera2048_no_errors.to_csv(no_errors_path)


#correct = chimera2048.loc[chimera2048["i"] == ground["instance"]]
"""