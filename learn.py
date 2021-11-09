import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from utils import add_neighbours, add_edges
from collections import namedtuple, deque
from data_gen import transform


# implement network as in article
class SGNN(nn.Module):

    def __init__(self):
        super(SGNN, self).__init__()

        self.edge1 = EdgeCentric(2, 2, 1, 2)  # edge [E, 4]
        self.node1 = NodeCentric(2, 2, 4, 5)  # node [N, 7]
        self.edge2 = EdgeCentric(7, 9, 4, 5)  # edge [E, 14]
        self.node2 = NodeCentric(7, 9, 14, 15)  # node [N, 24]

    def forward(self, data):
        data.edge_attr = F.relu(self.edge1(data))
        data.x = F.relu(self.node1(data))
        data.edge_attr = F.relu(self.edge2(data))
        data.x = F.relu(self.node2(data))

        return data


class DIRAC(nn.Module):

    def __init__(self):
        super(DIRAC, self).__init__()

        self.encoder = SGNN()
        self.fc1 = nn.Linear(24, 96)
        self.fc2 = nn.Linear(96, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, data, dim=2):
        # output should have size [N, 1] (Q-values)
        ising_graph = data
        data = transform(data, dim)
        data_encoded = self.encoder(data)
        Q = data_encoded.x  # node attributes
        Q = F.relu(self.fc1(Q))
        Q = F.relu(self.fc2(Q))
        Q = F.relu(self.fc3(Q))

        return Q


class EdgeCentric(nn.Module):

    def __init__(self, in_channels_x, out_channels_x, in_channels_e, out_channels_e):
        super(EdgeCentric, self).__init__()

        self.fcx = nn.Linear(in_channels_x, out_channels_x)
        self.fce = nn.Linear(in_channels_e, out_channels_e)

    def forward(self, data):
        # x has size [N, in_channels_x]
        # edge_index has size [2, E],
        # edge_attr has size [E, in_channels_e]
        # node_sum has size [E, num_of_node_features]
        # return has size [E, out_channels_x + out_channels_e]
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_attr = self.fce(edge_attr)
        node_sum = add_neighbours(x, edge_index)
        node_sum = self.fcx(node_sum)
        edge_attr = torch.cat((edge_attr, node_sum), dim=1)

        return edge_attr


class NodeCentric(nn.Module):

    def __init__(self, in_channels_x, out_channels_x, in_channels_e, out_channels_e):
        super(NodeCentric, self).__init__()

        self.fcx = nn.Linear(in_channels_x, out_channels_x)
        self.fce = nn.Linear(in_channels_e, out_channels_e)

    def forward(self, data):
        # x has size [N, in_channels_x]
        # edge_index has size [2, E],
        # edge_attr has size [E, in_channels_e]
        # edge_sum has size [N, num_of_edge_features]
        # return has size [N, out_channels_x + out_channels_e]
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fcx(x)
        edge_sum = add_edges(x, edge_index, edge_attr)
        edge_sum = self.fce(edge_sum)
        x = torch.cat((x, edge_sum), dim=1)
        return x


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






