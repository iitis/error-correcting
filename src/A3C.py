import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.DIRAC import SGNNMaxPool


class SharedAdam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
            weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq   `'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class SpinGlassActorCritic(nn.Module):
    def __init__(self, include_spin = False):
        super(SpinGlassActorCritic, self).__init__()

        encoder = SGNNMaxPool(include_spin=include_spin)

        self.fc1 = nn.Linear(12, 100)
        self.fc2 = nn.Linear(100, 25)
        self.fc3 = nn.Linear(25, 1)

    def forward(self, data):
        state_action_embedding = self.encoder(data.x, data.edge_index, data.edge_attr)
        z = state_action_embedding
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))

        pi = F.softmax(torch.reshape(self.fc3(z), (-1,))) # reshapes tensor from [N,1] int [N]

