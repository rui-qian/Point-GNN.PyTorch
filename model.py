import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, InstanceNorm1d


#############
# Point-GNN #
#############


class PointGNN(nn.Module):
    def __init__(self, kp_dim, state_dim, n_classes, n_iterations,
                 cls_channels=64, loc_channels=(64, 64, 7)):
        """
        Point-GNN. Network takes a dictionary containing a batch of graphs as
        well as the key-point features and the key-point lookup table
        for each vertex required for computing the initial vertex state.

        Args:
        :param kp_dim: number of key-point features
        :param state_dim: maximum number of state features
        :param n_classes: number of classes
        :param n_iterations: number of GNN iterations to perform
        :param cls_channels: number of channels in hidden layer of MLP for
                             classification
        :param loc_channels: number of channels of MLP for localization
        """
        super(PointGNN, self).__init__()

        # Set dimensionality of vertex state
        self._state_dim = state_dim

        # MLPs to compute initial vertex state from key-point features
        self._mlp_init = MLPInit(in_channel=kp_dim)
        self._mlp_init_aggr = MLPInitAggr()

        # List of GNN layers
        self._graph_layers = nn.ModuleList([StateConvLayer(state_dim) for _ in
                                           range(n_iterations)])

        # MLP for class prediction
        assert isinstance(cls_channels, int), 'Variable "cls_channels" must ' \
                                              'be an integer.'
        self._mlp_cls = Seq(basic_block(state_dim, cls_channels),
                            basic_block(cls_channels, n_classes))

        # Set of MLPs for per-class bounding box regression
        assert len(loc_channels) == 3, 'Tuple "loc_channels" must have a ' \
                                       'length of 3.'
        ch_l1, ch_l2, ch_l3 = loc_channels
        self._mlp_loc = nn.ModuleList([Seq(basic_block(state_dim, ch_l1),
                                           basic_block(ch_l1, ch_l2),
                                           basic_block(ch_l2, ch_l3)) for _ in
                                      range(n_classes)])

    def forward(self, batch_data):

        # Get data items
        batch_graph = batch_data['graphs']
        key_points = batch_data['key_points'].unsqueeze(0).float()
        key_points_lookup = batch_data['key_points_lookup'].long()

        # Get number of vertices in all graphs
        n_vertices = batch_graph.pos.size(0)

        # Container for initial vertex states
        initial_vertex_states = torch.zeros((n_vertices, self._state_dim))\
            .to(key_points.device)

        # Compute features for all key-points
        key_points_feats = self._mlp_init(key_points).squeeze(0)

        # Compute initial state for all vertices
        for vertex_id in range(n_vertices):

            # Get start and end indices for key-points
            start_id = key_points_lookup[vertex_id].item()
            try:
                end_id = key_points_lookup[vertex_id+1].item()
            except IndexError:
                end_id = key_points.size(1)

            # Get key-point features for currently inspected vertex
            vertex_kp_feats = key_points_feats[start_id:end_id]

            # Compute channel-wise max as initial vertex feature
            vertex_kp_feats_aggr = torch.max(vertex_kp_feats, dim=0)[0]

            # Apply MLP on aggregated features
            initial_vertex_state = self._mlp_init_aggr(vertex_kp_feats_aggr
                                                       .unsqueeze(0)
                                                       .unsqueeze(1))

            # Set state of currently inspected vertex
            initial_vertex_states[vertex_id, :] = initial_vertex_state

        # Set initial vertex state in graph
        batch_graph.x = initial_vertex_states

        # Get edge_index, vertex_state and vertex_coords for further processing
        # NOTE: s denotes vertex state and corresponds to x in the
        # PyTorch Geometric data object. Vertex position is denoted by x.
        edge_index, s, x = batch_graph.edge_index, batch_graph.x, \
                           batch_graph.pos

        # Perform GNN computations
        for graph_layer in self._graph_layers:
            # Update vertex state
            s = graph_layer(s, x, edge_index)

        # Compute classification and regression output for each vertex
        cls_pred = self._mlp_cls(s).squeeze(0)
        reg_pred = torch.cat([mlp_loc(s) for mlp_loc in self._mlp_loc],
                             dim=2).squeeze(0)

        return cls_pred, reg_pred


class StateConvLayer(MessagePassing):
    def __init__(self, state_dim, channels_h=(64, 3)):
        """
        Graph Layer to perform update of the vertex states.
        """
        super(StateConvLayer, self).__init__(aggr='max',
                                             flow='target_to_source')

        # MLP to transform vertex state into auto-registration offset
        assert len(channels_h) == 2, 'Tuple "channels_h" must have a length ' \
                                     'of 2.'
        ch_h1, ch_h2 = channels_h
        self._mlp_h = Seq(basic_block(state_dim, ch_h1),
                          basic_block(ch_h1, ch_h2))

        # MLP to compute edge features
        self._mlp_f = Seq(basic_block(state_dim+3, state_dim),
                          basic_block(state_dim, state_dim))
        self._mlp_g = Seq(basic_block(state_dim, state_dim),
                          basic_block(state_dim, state_dim))

    def forward(self, s, x, edge_index):
        return self.propagate(edge_index, s=s, x=x)

    def message(self, x_j, x_i, s_i, s_j):

        # NOTE: Unsqueezing first dimension required for computation of
        # Instance Normalization in MLPs

        # Compute auto-registration offset
        delta_x_i = self._mlp_h(s_i.unsqueeze(0))

        # Compute edge features
        tmp = torch.cat([x_j.unsqueeze(0) - x_i.unsqueeze(0) -
                         delta_x_i, s_j.unsqueeze(0)], dim=2)
        e_ij = self._mlp_f(tmp).squeeze(0)
        return e_ij

    def update(self, e_ij, s):
        # Update vertex state based on aggregated edge features
        return s.unsqueeze(0) + self._mlp_g(e_ij.unsqueeze(0))


def basic_block(in_channel, out_channel):
    """
    Create block with linear layer followed by IN and ReLU.
    :param in_channel: number of input features
    :param out_channel: number of output features
    :return: PyTorch Sequential object
    """
    return Seq(Linear(in_channel, out_channel), InstanceNorm1d(out_channel),
               ReLU(inplace=True))


class MLPInit(nn.Module):
    def __init__(self, in_channel, channels=(32, 64, 128, 300)):
        """
        MLP to transform initial key-point features into embeddings.
        """
        super(MLPInit, self).__init__()

        # Check validity of provided channels
        assert len(channels) == 4, 'Tuple "channels" must have a length of 4.'
        ch1, ch2, ch3, ch4 = channels

        # Create network blocks
        self._block1 = basic_block(in_channel, ch1)
        self._block2 = basic_block(ch1, ch2)
        self._block3 = basic_block(ch2, ch3)
        self._block4 = basic_block(ch3, ch4)

    def forward(self, x):
        out = self._block1(x)
        out = self._block2(out)
        out = self._block3(out)
        out = self._block4(out)
        return out


class MLPInitAggr(nn.Module):
    def __init__(self, state_dim=300):
        """
        MLP to extract features from aggregated key-point features.
        """
        super(MLPInitAggr, self).__init__()
        self._block1 = basic_block(state_dim, state_dim)
        self._block2 = basic_block(state_dim, state_dim)

    def forward(self, x):
        out = self._block1(x)
        out = self._block2(out)
        return out

