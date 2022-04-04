import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.nn import DataParallel, GAE
from src.DIRAC import SGNNMaxPool, SGNN
from src.data_gen import generate_chimera

chimera = generate_chimera(3,3)
encoder = SGNNMaxPool(include_spin=True)

class SGDecoder(nn.Module):
    def __init__(self):
        super(SGDecoder, self).__init__()


    def forward(self, z):
        pass

class SGGAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(SGGAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        z = self.encode(x, edge_index, edge_attr)
        enc_x, enc_edge_index, enc_edge_attr = self.decode(z)

        return enc_x, enc_edge_index, enc_edge_attr