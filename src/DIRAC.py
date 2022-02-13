
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch import Tensor, LongTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# implement network as in article
class DIRAC(nn.Module):

    def __init__(self, include_spin=False):
        super(DIRAC, self).__init__()

        self.encoder = SGNNMaxPool(include_spin=include_spin)
        #self.encoder = nn.DataParallel(self.encoder)

        self.fc1 = nn.Linear(12, 36)
        init.xavier_normal_(self.fc1.weight, gain=init.calculate_gain('relu'))

        self.fc2 = nn.Linear(36, 10)
        init.xavier_normal_(self.fc2.weight, gain=init.calculate_gain('relu'))

        self.fc3 = nn.Linear(10, 1)
        init.xavier_normal_(self.fc3.weight, gain=init.calculate_gain('leaky_relu', 0.2))


    def forward(self, batch):
        # output should have size [1, N] (Q-values)

        state_action_embedding = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        Q = state_action_embedding
        Q = F.relu(self.fc1(Q))
        Q = F.dropout(Q, p=0.5, training=self.training)

        Q = F.relu(self.fc2(Q))
        Q = F.dropout(Q, p=0.5, training=self.training)

        Q = F.leaky_relu(self.fc3(Q), 0.2)
        return Q.reshape(-1)  # vector [1,N]


"""
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
"""

class EdgeCentric(nn.Module):

    def __init__(self, in_channels_x: int, out_channels_x: int, in_channels_e: int, out_channels_e: int) -> None:
        super(EdgeCentric, self).__init__()

        self.fcx = nn.Linear(in_channels_x, out_channels_x)
        self.fce = nn.Linear(in_channels_e, out_channels_e)

    def forward(self, x, edge_index, edge_attr):

        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]
        x = x_i + x_j
        x = self.fcx(x)
        edge_attr = self.fce(edge_attr)
        output = torch.cat((x, edge_attr), dim=1)
        return output


class NodeCentric(nn.Module):
    def __init__(self, in_channels_x: int, out_channels_x: int, in_channels_e: int, out_channels_e: int) -> None:
        super(NodeCentric, self).__init__()

        self.fcx = nn.Linear(in_channels_x, out_channels_x)
        self.fce = nn.Linear(in_channels_e, out_channels_e)

    def forward(self, x, edge_index, edge_attr):

        adj = torch.sparse_coo_tensor(edge_index, edge_attr)
        adj = adj.to_dense()
        adj = torch.sum(adj, dim=1)
        x = self.fcx(x)
        edge_attr = self.fce(adj)
        output = torch.cat((x, edge_attr), dim=1)
        return output


class SGNNMaxPool(nn.Module):

    def __init__(self, include_spin: bool = False):
        super(SGNNMaxPool, self).__init__()

        self.max_pool = nn.MaxPool1d(3)

        if include_spin:
            self.edge1 = EdgeCentric(6, 10, 1, 2)  # Edge 1->12
             # Edge 12->4
            self.node1 = NodeCentric(6, 3, 4, 3)  # Node 6->30
              # Node 6->6
        else:
            self.edge1 = EdgeCentric(5, 10, 1, 2)  # Edge 1->12
             # Edge 12->4
            self.node1 = NodeCentric(5, 3, 4, 3)  # Node 5->30
              # Node 6->6

        self.edge2 = EdgeCentric(6, 18, 4, 12)  # Edge 4->30
         # Edge 30 -> 10
        self.node2 = NodeCentric(6, 3, 10, 3)  # Node 10->30
          # Node 30->10

        self.edge3 = EdgeCentric(6, 18, 10, 12)  # Edge 10->30
         # Edge 30 -> 10
        self.node3 = NodeCentric(6, 3, 10, 3)  # Node 10->30
          # Node 30->10
        """
        self.edge4 = EdgeCentric(10, 18, 10, 12)  # Edge 10->30
         # Edge 30 -> 10
        self.node4 = NodeCentric(10, 18, 10, 12)  # Node 10->30
          # Node 30->10

        self.edge5 = EdgeCentric(10, 3, 10, 3)  # Edge 10->6
        self.node5 = NodeCentric(10, 3, 6, 3)  # Node 10->6
"""

    def forward(self, x: Tensor, edge_index: LongTensor, edge_attr: Tensor) -> Tensor:

        edge_attr = self.max_pool(F.relu(self.edge1(x, edge_index, edge_attr)))
        x = F.relu(self.node1(x, edge_index, edge_attr))

        edge_attr = self.max_pool(F.relu(self.edge2(x, edge_index, edge_attr)))
        x = F.relu(self.node2(x, edge_index, edge_attr))

        edge_attr = self.max_pool(F.relu(self.edge3(x, edge_index, edge_attr)))
        x = F.leaky_relu(self.node3(x, edge_index, edge_attr), 0.2)
        """
        edge_attr = self.max_pool(F.relu(self.edge4(x, edge_index, edge_attr)))
        x = self.max_pool(F.relu(self.node4(x, edge_index, edge_attr)))

        edge_attr = F.relu(self.edge5(x, edge_index, edge_attr))
        x = F.relu(self.node5(x, edge_index, edge_attr))
        """
        action_embedding = x  # node attributes [N, 6]
        state_embedding = torch.sum(action_embedding, dim=0)  # [1, 6]
        # add virtual empty first dimensions (in pytorch size [1,6] is just [6])
        state_embedding = state_embedding[None, :]
        N = x.size()[0]
        # change od dimensions to later concatenate. Repeat N times along Y axis and once along X axis (so we got [N,6])
        state_embedding = state_embedding.repeat(N, 1)  # change od dimensions to later concatenate

        output = torch.cat((state_embedding, action_embedding), dim=1)  # [N, 12]

        return output