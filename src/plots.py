"""
Simple script for generating plots etc.
"""

import torch
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

def correction(x):
    return x+75

chimera2048_path = "/home/tsmierzchalski/pycharm_projects/error-correcting/chimera2048.csv"
ground_path = "/home/tsmierzchalski/pycharm_projects/error-correcting/groundstates_otn2d.csv"
no_errors_path = "/home/tsmierzchalski/pycharm_projects/error-correcting/chimera2048_no_errors.csv"
errors_path = "/home/tsmierzchalski/pycharm_projects/error-correcting/errors.csv"
correct_path = "/home/tsmierzchalski/pycharm_projects/error-correcting/chimera2048_approx_correct.csv"

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