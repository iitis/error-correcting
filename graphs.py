import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

energy_trained_graphs = "/home/tsmierzchalski/pycharm_projects/error-correcting/energy_trained_graphs.csv"

df = pd.read_csv(energy_trained_graphs)

def f(x):
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 225
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * x / EPS_DECAY)

t1 = np.arange(0.0, 1000.0, 1.0)
y = [f(x) for x in t1]
plt.plot(t1, y)
plt.show()
"""
for i in range(1, 10):
    graph = df.iloc[:, i]

    graph.plot()
    plt.show()
"""