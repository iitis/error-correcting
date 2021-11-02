import torch.nn as nn
from torch_geometric.nn import MessagePassing

# implement network as in article
class SGNN(nn.Module):
    def __init__(self):
        super(SGNN, self).__init__()

    def forward(self, x):
        pass


class EdgeCentric(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeCentric).__init__()


class NodeCentric(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(NodeCentric).__init__()

        self.relu = nn.ReLU()

    def message(self, x_j):
        pass

    def propagate(self, edge_index, size):
        pass