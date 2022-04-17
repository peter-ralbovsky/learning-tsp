import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

def get_edge_features(x):
    num_nodes = x.shape[1]
    new_size = [-1 for _ in range(len(x.shape)+1)]
    new_size[1] = num_nodes
    e_from =x.unsqueeze(1).expand(*new_size)
    new_size[1] = -1
    new_size[2] = num_nodes
    e_to = x.unsqueeze(2).expand(*new_size)
    return e_from, e_to

def get_pairwise_dists(x):
    f, t = get_edge_features(x)
    return torch.sqrt(torch.sum((f-t)**2, dim=-1, keepdim=True))



class EGNNLayer(nn.Module):
    """Configurable GNN Layer

    Implements the Gated Graph ConvNet layer combined with equivariant graph neural network:
        h_i = ReLU ( U*h_i + Aggr.( sigma_ij, V*h_j) ),
        x_i = x_i + c \sum_j\neq i (x_i -x_j sigma(e*(e_ij)))
        sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij ),
        e_ij = ReLU ( A*h_i + B*h_j + C*e_ij + d*|x_i-x_j|),
        where Aggr. is an aggregation function: sum/mean/max.

    References:
        - X. Bresson and T. Laurent. An experimental study of neural networks for variable graphs. In International Conference on Learning Representations, 2018.
        - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.
    """

    def __init__(self, hidden_dim, aggregation="sum", norm="batch", learn_norm=True, track_norm=False, gated=True):
        """
        Args:
            hidden_dim: Hidden dimension size (int)
            aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
            norm: Feature normalization scheme ("layer"/"batch"/None)
            learn_norm: Whether the normalizer has learnable affine parameters (True/False)
            track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
            gated: Whether to use edge gating (True/False)
        """
        super(EGNNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.norm = norm
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        self.gated = gated
        assert self.gated, "Use gating with GCN, pass the `--gated` flag"
        
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.c = nn.Parameter(torch.Tensor(1))
        self.d = nn.Linear(1,hidden_dim)
        self.e = nn.Linear(hidden_dim, 1)

        self.norm_h = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)

        self.norm_e = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)
        
    def forward(self, h, e, x, graph):
        """
        Args:
            h: Input node features (B x V x H)
            e: Input edge features (B x V x V x H)
            x: Input node coordinates (B x V x 2)
            graph: Graph adjacency matrices (B x V x V)
        Returns: 
            Updated node and edge features
        """
        batch_size, num_nodes, hidden_dim = h.shape
        h_in = h
        e_in = e
        x_in = x

        # Linear transformations for node update
        Uh = self.U(h)  # B x V x H
        Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H

        # Linear transformations for edge update and gating
        Ah = self.A(h)  # B x V x H
        Bh = self.B(h)  # B x V x H
        Ce = self.C(e)  # B x V x V x H

        distances = get_pairwise_dists(x_in)

        # Update edge features and compute edge gates
        e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce + self.d(distances)  # B x V x V x H
        gates = torch.sigmoid(e)  # B x V x V x H

        x_from, x_to = get_edge_features(x_in)


        point_attention = torch.sigmoid(self.e(e))
        x = self.c*torch.sum(point_attention * (x_to - x_from), dim = -2)



        # Update node features
        h = Uh + self.aggregate(Vh, graph, gates)  # B x V x H


        # Normalize node features
        h = self.norm_h(
            h.view(batch_size*num_nodes, hidden_dim)
        ).view(batch_size, num_nodes, hidden_dim) if self.norm_h else h
        
        # Normalize edge features
        e = self.norm_e(
            e.view(batch_size*num_nodes*num_nodes, hidden_dim)
        ).view(batch_size, num_nodes, num_nodes, hidden_dim) if self.norm_e else e

        # Apply non-linearity
        h = F.relu(h)
        e = F.relu(e)

        # Make residual connection
        h = h_in + h
        e = e_in + e
        x = x_in + x

        return h, e, x

    def aggregate(self, Vh, graph, gates):
        """
        Args:
            Vh: Neighborhood features (B x V x V x H)
            graph: Graph adjacency matrices (B x V x V)
            gates: Edge gates (B x V x V x H)
        Returns:
            Aggregated neighborhood features (B x V x H)
        """
        # Perform feature-wise gating mechanism
        Vh = gates * Vh  # B x V x V x H
        
        # Enforce graph structure through masking
        Vh[graph.unsqueeze(-1).expand_as(Vh)] = 0
        
        if self.aggregation == "mean":
            return torch.sum(Vh, dim=2) / torch.sum(1-graph, dim=2).unsqueeze(-1).type_as(Vh)
        
        elif self.aggregation == "max":
            return torch.max(Vh, dim=2)[0]
        
        else:
            return torch.sum(Vh, dim=2)
        

class EGNNEncoder(nn.Module):
    """Configurable GNN Encoder
    """
    
    def __init__(self, n_layers, hidden_dim, aggregation="sum", norm="layer", 
                 learn_norm=True, track_norm=False, gated=True, *args, **kwargs):
        super(EGNNEncoder, self).__init__()

        self.init_embed_edges = nn.Embedding(2, hidden_dim)

        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
                for _ in range(n_layers)
        ])

    def forward(self, x, graph):
        """
        Args:
            x: Input node features (B x V x H)
            graph: Graph adjacency matrices (B x V x V)
        Returns: 
            Updated node features (B x V x H)
        """
        # Embed edge features
        e = self.init_embed_edges(graph.type(torch.long))
        h = x
        for layer in self.layers:
            h, e, x = layer(h, e, x, graph)

        return h
