import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
import torch_geometric as tg

class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input, *args, **kwargs):
        return input + self.module(input, *args, **kwargs)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim, val_dim=None, key_dim=None):
        super(MultiHeadAttention, self).__init__()
        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = nn.Linear(input_dim*n_heads, key_dim*n_heads)
        self.W_key = nn.Linear(input_dim*n_heads, key_dim*n_heads)
        self.W_val = nn.Linear(input_dim*n_heads, val_dim*n_heads)

        if embed_dim is not None:
            self.W_out = nn.Linear(torch.Tensor(n_heads*key_dim, embed_dim))

        self.init_parameters()