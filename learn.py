import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from utils import add_neighbours, add_edges


# implement network as in article
class SGNN(nn.Module):
    def __init__(self):
        super(SGNN, self).__init__()
        self.edge1 = EdgeCentric(2, 4, 1, 2)
        self.node1 = NodeCentric(2, 4, 6, 10)

    def forward(self, data):
        data.edge_attr = F.relu(self.edge1(data))
        data.x = F.relu(self.node1(data))

        return data


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









