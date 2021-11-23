import torch
import math
import random as rn
import torch.optim as optim
from enviroment import IsingGraph2dRandom
from data_gen import generate_dataset
from learn import DIRAC, ReplayMemory
from utils import compute_energy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

policy_net = DIRAC()
target_net = DIRAC()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters())

#dataset = generate_dataset(100, 10)
env = IsingGraph2dRandom((2, 2))

memory = ReplayMemory(10000)
n_actions = env.action_space.n
steps_done = 0

print(policy_net(env.state).argmax().item())

_, reward, _, _ = env.step(1)
print(reward)
env.render()


def select_action(state):
    global steps_done
    sample = rn.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)  # epsilon decay
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            pass

