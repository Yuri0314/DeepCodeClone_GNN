import dgl
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import GATConv
from dgl.nn import GlobalAttentionPooling


class GAT(nn.Module):
    def __init__(self,
                 tokens_size,
                 in_dim,
                 hidden_dims,
                 out_dim,
                 num_heads,
                 num_layers=4,
                 feat_drop=.2,
                 attn_drop=.2,
                 negative_slope=.2,
                 residual=False,
                 activation=F.relu):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        # input embedding
        self.embed = nn.Embedding(tokens_size, in_dim)
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, hidden_dims[0], num_heads[0],
            feat_drop, attn_drop, negative_slope, False, activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                hidden_dims[l - 1] * num_heads[l - 1], hidden_dims[l], num_heads[l],
                feat_drop, attn_drop, negative_slope, residual, activation))
        # readout funtion
        self.readout_layer = GlobalAttentionPooling(nn.Sequential(nn.Linear(out_dim, 1), nn.Sigmoid()))

    def forward(self, inputs):
        x, edges, edge_types = inputs
        # input embedding
        h = self.embed(x).squeeze(1)
        u, v = zip(*edges)
        g = dgl.graph((u, v))
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # readout funtion
        graph_vector = self.readout_layer(g, h)
        return graph_vector
