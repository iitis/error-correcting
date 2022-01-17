
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# implement network as in article
class SGNN(nn.Module):

    def __init__(self, include_spin=False):
        super(SGNN, self).__init__()
        if include_spin:
            self.edge1 = EdgeCentric(6, 6, 1, 2)  # Edge 1->8
            self.node1 = NodeCentric(6, 6, 8, 10)  # Node 4->16
        else:
            self.edge1 = EdgeCentric(5, 6, 1, 2)  # Edge 1->8
            self.node1 = NodeCentric(5, 6, 8, 10)  # Node 4->16

        self.edge2 = EdgeCentric(16, 32, 8, 16)  # Edge 8->48
        self.node2 = NodeCentric(16, 32, 48, 24)  # Node 16->56

        self.edge3 = EdgeCentric(56, 24, 48, 24)  # Edge 48->48
        self.node3 = NodeCentric(56, 24, 48, 24)  # Node 56->48

        self.edge4 = EdgeCentric(48, 13, 48, 13)  # Edge 48->26
        self.node4 = NodeCentric(48, 13, 26, 8)  # Node 48->21

        self.edge5 = EdgeCentric(21, 3, 26, 3)  # Edge 26->6
        self.node5 = NodeCentric(21, 3, 6, 3)  # Node 21->6

    def forward(self, x, edge_index, edge_attr):

        edge_attr = F.relu(self.edge1(x, edge_index, edge_attr))
        x = F.relu(self.node1(x, edge_index, edge_attr))

        edge_attr = F.relu(self.edge2(x, edge_index, edge_attr))
        x = F.relu(self.node2(x, edge_index, edge_attr))

        edge_attr = F.relu(self.edge3(x, edge_index, edge_attr))
        x = F.relu(self.node3(x, edge_index, edge_attr))

        edge_attr = F.relu(self.edge4(x, edge_index, edge_attr))
        x = F.relu(self.node4(x, edge_index, edge_attr))

        edge_attr = F.relu(self.edge5(x, edge_index, edge_attr))
        x = F.relu(self.node5(x, edge_index, edge_attr))

        action_embedding = x  # node attributes [N, 6]
        state_embedding = torch.sum(action_embedding, dim=0)  # [1, 6]
        # add virtual empty first dimensions (in pytorch size [1,6] is just [6])
        state_embedding = state_embedding[None, :]
        state_embedding = state_embedding.repeat(x.size()[0], 1)  # change od dimensions to later concatenate

        output = torch.cat((state_embedding, action_embedding), dim=1)  # [N, 12]

        return output


class DIRAC(nn.Module):

    def __init__(self, include_spin=False):
        super(DIRAC, self).__init__()

        self.encoder = SGNNMaxPool(include_spin=include_spin).to(device)

        self.fc1 = nn.Linear(12, 100)
        self.fc2 = nn.Linear(100, 25)
        self.fc3 = nn.Linear(25, 1)

    def forward(self, batch):
        # output should have size [1, N] (Q-values)

        state_action_embedding = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        Q = state_action_embedding  # change matrix [N+1, 6] into vector
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

        return torch.cat((x, edge_attr), dim=1)


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

        return torch.cat((x, edge_attr), dim=1)


class SGNNMaxPool(nn.Module):

    def __init__(self, include_spin=False):
        super(SGNNMaxPool, self).__init__()

        self.max_pool = nn.MaxPool1d(3)

        if include_spin:
            self.edge1 = EdgeCentric(6, 10, 1, 2)  # Edge 1->12
             # Edge 12->4
            self.node1 = NodeCentric(6, 18, 4, 12)  # Node 6->30
              # Node 30->10
        else:
            self.edge1 = EdgeCentric(5, 10, 1, 2)  # Edge 1->12
             # Edge 12->4
            self.node1 = NodeCentric(5, 18, 4, 12)  # Node 5->30
              # Node 30->10

        self.edge2 = EdgeCentric(10, 18, 4, 12)  # Edge 4->30
         # Edge 30 -> 10
        self.node2 = NodeCentric(10, 18, 10, 12)  # Node 10->30
          # Node 30->10

        self.edge3 = EdgeCentric(10, 18, 10, 12)  # Edge 10->30
         # Edge 30 -> 10
        self.node3 = NodeCentric(10, 18, 10, 12)  # Node 10->30
          # Node 30->10

        self.edge4 = EdgeCentric(10, 18, 10, 12)  # Edge 10->30
         # Edge 30 -> 10
        self.node4 = NodeCentric(10, 18, 10, 12)  # Node 10->30
          # Node 30->10

        self.edge5 = EdgeCentric(10, 3, 10, 3)  # Edge 10->6
        self.node5 = NodeCentric(10, 3, 6, 3)  # Node 10->6


    def forward(self, x, edge_index, edge_attr):

        edge_attr = self.max_pool(F.relu(self.edge1(x, edge_index, edge_attr)))
        x = self.max_pool(F.relu(self.node1(x, edge_index, edge_attr)))

        edge_attr = self.max_pool(F.relu(self.edge2(x, edge_index, edge_attr)))
        x = self.max_pool(F.relu(self.node2(x, edge_index, edge_attr)))

        edge_attr = self.max_pool(F.relu(self.edge3(x, edge_index, edge_attr)))
        x = self.max_pool(F.relu(self.node3(x, edge_index, edge_attr)))

        edge_attr = self.max_pool(F.relu(self.edge4(x, edge_index, edge_attr)))
        x = self.max_pool(F.relu(self.node4(x, edge_index, edge_attr)))

        edge_attr = F.relu(self.edge5(x, edge_index, edge_attr))
        x = F.relu(self.node5(x, edge_index, edge_attr))

        action_embedding = x  # node attributes [N, 6]
        state_embedding = torch.sum(action_embedding, dim=0)  # [1, 6]
        # add virtual empty first dimensions (in pytorch size [1,6] is just [6])
        state_embedding = state_embedding[None, :]
        state_embedding = state_embedding.repeat(x.size()[0], 1)  # change od dimensions to later concatenate

        output = torch.cat((state_embedding, action_embedding), dim=1)  # [N, 12]

        return output



