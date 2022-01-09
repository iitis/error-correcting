import pandas as pd
import pickle

GROUND = "/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/groundstates_otn2d.txt"

ground = pd.read_csv(GROUND, delimiter=" ", header=None)
ground.drop(1, inplace=True, axis=1)

solution_dict = {}
for index, row in ground.iterrows():
    spins = {i-3: 1 if row[i] == 1 else -1 for i in range(3, len(row)+1)}
    sol = {"ground": row[2], "spins": spins}
    solution_dict["{}".format(index + 1)] = sol

with open('/home/tsmierzchalski/pycharm_projects/error-correcting/datasets/chimera_512_sol.pkl', 'wb') as f:
    pickle.dump(solution_dict, f)




