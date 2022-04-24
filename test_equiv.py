import torch
from nets.encoders.egnn_encoder import EGNNLayer
from scipy.stats import ortho_group
def random_orthogonal_matrix(dim=3):
  """Helper function to build a random orthogonal matrix of shape (dim, dim)
  """
  Q = torch.tensor(ortho_group.rvs(dim=dim)).float()
  return Q

def rot_trans_equivariance_unit_test(module):
    """Unit test for checking whether a module (GNN layer) is
    rotation and translation equivariant.
    """
    nu = """
        Args:
            h: Input node features (B x V x H)
            e: Input edge features (B x V x V x H)
            x: Input node coordinates (B x V x hp x 2)
            graph: Graph adjacency matrices (B x V x V)
        Returns: 
            Updated node and edge features
    """

    B = 128
    V = 20
    H = 128
    hp = 3
    h = torch.rand((B, V, H))
    e = torch.rand((B, V, V, H))
    x = torch.rand((B, V, hp, 2))
    graph = torch.zeros((B, V, V)).bool()


    h_1, e_1, x_1 = module(h, e, x, graph)

    Q = random_orthogonal_matrix(dim=2)
    t = torch.rand(2)
    # ============ YOUR CODE HERE ==============
    # Perform random rotation + translation on data.
    #
    print(x.shape)
    x = x@(Q.T)+ t
    print(x.shape)
    # ==========================================

    # Forward pass on rotated + translated example
    h_2, e_2, x_2 = module(h, e, x, graph)

    # ============ YOUR CODE HERE ==============
    # Check whether output varies after applying transformations.
    print(torch.allclose(x_1@(Q.T)+ t, x_2, atol=1e-04))
    print(torch.allclose(h_1 , h_2, atol=1e-04))
    print(torch.allclose(e_1, e_2, atol=1e-04))
    # ==========================================

model = EGNNLayer(hidden_dim=128, aggregation="max", norm="batch", learn_norm=True, track_norm=False, gated=True, num_coordinates=3)
rot_trans_equivariance_unit_test(model)

