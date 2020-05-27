import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, InstanceNorm1d


###############
# Graph Layer #
###############

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


###############
# Basic Block #
###############

def basic_block(in_channel, out_channel):
    """
    Create block with linear layer followed by IN and ReLU.
    :param in_channel: number of input features
    :param out_channel: number of output features
    :return: PyTorch Sequential object
    """
    return Seq(Linear(in_channel, out_channel), InstanceNorm1d(out_channel),
               ReLU(inplace=True))


#############
# Init MLPs #
#############

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

