from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU
import torch
import torch.nn as nn


class PointGNN(nn.Module):
    def __init__(self, kp_dim, state_dim, n_classes, n_iterations):
        """
        Point-GNN. Network takes a dictionary containing a batch of graphs as
        well as the key-point features for each vertex required for computing
        the initial vertex state.

        Args:
        :param kp_dim: number of key-point features
        :param state_dim: maximum number of state features
        :param n_classes: number of classes
        :param n_iterations: number of GNN iterations to perform
        """
        super(PointGNN, self).__init__()

        self.state_dim = state_dim
        self.n_classes = n_classes
        self.n_iterations = n_iterations

        # MLP to compute initial vertex state from key-point features
        self.mlp_init = MLPinit(in_channel=kp_dim)

        # List of GNN layers
        self.graph_layers = nn.ModuleList([StateConvLayer(state_dim) for _ in
                                     range(n_iterations)])

        # MLP for class prediction
        self.mlp_class = Seq(basic_block(state_dim, 64),
                             basic_block(64, n_classes))

        # Set of MLPs for per-class bounding box regression
        self.mlp_loc = nn.ModuleList([Seq(basic_block(state_dim, 64),
                                          basic_block(64, 64),
                                          basic_block(64, 7)) for _ in
                                      range(n_classes)])

    def forward(self, batch_data):

        graph = batch_data['data']
        key_points = batch_data['key_points']

        # Compute initial vertex state
        # TODO: Fix batching
        vertex_features = torch.zeros((len(key_points), self.state_dim))
        for key_point_id, key_point in enumerate(key_points):
            initial_vertex_state = torch.max(self.mlp_init(key_point
                                                           .unsqueeze(0)
                                                           .float()),
                                             dim=1)[0]
            vertex_features[key_point_id, :] = initial_vertex_state[0]

        # Set initial vertex state
        graph.x = vertex_features

        # Get edge_index, vertex_state and vertex_coords for further processing
        edge_index, s, x = graph.edge_index, graph.x, graph.pos

        # Perform GNN computations
        for graph_layer in self.graph_layers:
            # Update vertex state
            s = graph_layer(s, x, edge_index)

        s = s.unsqueeze(0)  # quick fix until batching is done properly

        # Compute classification and regression output for each vertex
        cls_pred = self.mlp_class(s)
        reg_pred = torch.cat([mlp_loc(s) for mlp_loc in self.mlp_loc], dim=2)

        return cls_pred, reg_pred


class StateConvLayer(MessagePassing):
    def __init__(self, state_dim):
        """
        Graph Layer to perform update of the vertex states.
        """
        super(StateConvLayer, self).__init__(aggr='max')

        # MLP to transform vertex state into auto-registration offset
        self.mlp_h = Seq(Linear(state_dim, 64), Linear(64, 3))

        # MLP to compute edge features
        self.mlp_f = Seq(Linear(state_dim+3, 300), Linear(300, 300))
        self.mlp_g = Seq(Linear(state_dim, state_dim), Linear(state_dim,
                                                              state_dim))

    def forward(self, s, x, edge_index):
        return self.propagate(edge_index, s=s, x=x)

    def message(self, x_j, x_i, s_i, s_j):

        # Compute auto-registration offset
        delta_x_i = self.mlp_h(s_i)

        # Compute edge features
        tmp = torch.cat([x_j - x_i - delta_x_i, s_j], dim=1)
        e_ij = self.mlp_f(tmp)
        return e_ij

    def update(self, e_ij, s):
        # Update vertex state based on aggregated edge features
        return s + self.mlp_g(e_ij)


def basic_block(in_channel, out_channel):
    """
    Create block with linear layer followed by IN and ReLU.
    :param in_channel: number of input features
    :param out_channel: number of output features
    :return: PyTorch Sequential object
    """
    return nn.Sequential(Linear(in_channel, out_channel),
                         nn.InstanceNorm1d(out_channel),
                         nn.ReLU(inplace=True))


class MLPinit(nn.Module):
    def __init__(self, in_channel):
        """
        MLP to transform initial key-point features into embeddings.
        """
        super(MLPinit, self).__init__()
        self.block1 = basic_block(in_channel, 32)
        self.block2 = basic_block(32, 64)
        self.block3 = basic_block(64, 128)
        self.block4 = basic_block(128, 300)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        return out
