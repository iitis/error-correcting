import torch.optim as optim
from enviroment import IsingGraph2d
from data_gen import generate_dataset
from learn import DIRAC, ReplayMemory

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

dataset = generate_dataset(10, 1)
env = IsingGraph2d(dataset)

memory = ReplayMemory(10)
n_actions = env.action_space.n
steps_done = 0


