import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch


class Encoder(MessagePassing):
    def __init__(self):
        super().__init__()

