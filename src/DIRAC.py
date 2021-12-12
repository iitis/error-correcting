import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from collections import namedtuple, deque
from src.data_gen import transform, transform_batch_square
from torch_sparse import SparseTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# implement network as in article
class SGNN(nn.Module):

    def __init__(self):
        super(SGNN, self).__init__()

        self.edge1 = EdgeCentric(2, 2, 1, 2)  # edge [E, 4]
        self.node1 = NodeCentric(2, 2, 4, 5)  # node [N, 7]
        self.edge2 = EdgeCentric(7, 9, 4, 5)  # edge [E, 14]
        self.node2 = NodeCentric(7, 9, 14, 15)  # node [N, 24]
        self.edge3 = EdgeCentric(24, 30, 14, 15)  # edge [E, 45]
        self.node3 = NodeCentric(24, 30, 45, 15)  # node [N, 45]
        self.edge4 = EdgeCentric(45, 20, 45, 15)  # edge [E, 35]
        self.node4 = NodeCentric(45, 20, 35, 10)  # node [N, 30]
        self.edge5 = EdgeCentric(30, 3, 35, 2)  # edge [E, 5]
        self.node5 = NodeCentric(30, 3, 5, 2)  # node [N, 5]

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x.to(device)
        edge_index.to(device)
        edge_attr.to(device)
        data.cuda()

        data.edge_attr = F.relu(self.edge1(data))
        data.x = F.relu(self.node1(data))
        data.edge_attr = F.relu(self.edge2(data))
        data.x = F.relu(self.node2(data))
        data.edge_attr = F.relu(self.edge3(data))
        data.x = F.relu(self.node3(data))
        data.edge_attr = F.relu(self.edge4(data))
        data.x = F.relu(self.node4(data))
        data.edge_attr = F.relu(self.edge5(data))
        data.x = F.relu(self.node5(data))

        action_embedding = data.x  # node attributes [N, 5]
        state_embedding = torch.sum(action_embedding, dim=0)  # [1, 5]
        state_embedding = state_embedding[None, :]
        state_embedding = state_embedding.repeat(data.x.size()[0], 1)  # change od dimenstions to later concanate

        output = torch.cat((action_embedding, state_embedding), dim=1)  # [N, 10]

        return output


class DIRAC(nn.Module):

    def __init__(self, dim=(3, 3)):
        super(DIRAC, self).__init__()

        self.dim = dim
        self.encoder = SGNN().to(device)

        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 25)
        self.fc3 = nn.Linear(25, 1)

    def forward(self, batch):
        # output should have size [1, N] (Q-values)
        if isinstance(batch, Batch):
            batch = transform_batch_square(batch)
        else:
            batch = transform(batch, len(self.dim))

        state_action_embedding = self.encoder(batch)
        Q = state_action_embedding  # change matrix [N+1, 5] into vector
        Q = F.relu(self.fc1(Q))
        Q = F.relu(self.fc2(Q))
        Q = F.relu(self.fc3(Q))

        return Q.reshape(-1)


class EdgeCentric(nn.Module):

    def __init__(self, in_channels_x, out_channels_x, in_channels_e, out_channels_e):
        super(EdgeCentric, self).__init__()

        self.fcx = nn.Linear(in_channels_x, out_channels_x)
        self.fce = nn.Linear(in_channels_e, out_channels_e)

    def forward(self, x, edge_index, edge_attr):

        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]
        x = x_i + x_j
        x = self.fcx(x)
        edge_attr = self.fce(edge_attr)

        return F.relu(torch.cat((x, edge_attr), dim=1))


class NodeCentric(nn.Module):
    def __init__(self, in_channels_x, out_channels_x, in_channels_e, out_channels_e):
        super(NodeCentric, self).__init__()

        self.fcx = nn.Linear(in_channels_x, out_channels_x)
        self.fce = nn.Linear(in_channels_e, out_channels_e)

    def forward(self, x, edge_index, edge_attr):

        adj = torch.sparse_coo_tensor(edge_index, edge_attr)
        adj = adj.to_dense()
        adj = torch.sum(adj, dim=1)
        x = self.fcx(x)
        edge_attr = self.fce(adj)

        return F.relu(torch.cat((x, edge_attr), dim=1))


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


