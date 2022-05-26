import torch.nn as nn
import torch.nn.functional as F

from model.clone_detection_DNN import DNN
from model.gat_net import GAT


class DCC_GNN(nn.Module):
    def __init__(self,
                 tokens_size,
                 in_dim,
                 GNN_hidden_dims,
                 graph_dim,
                 num_heads,
                 DNN_hidden_dims1,
                 DNN_hidden_dims2,
                 class_num=2,
                 GNN_layers=4,
                 feat_drop=.2,
                 attn_drop=.2,
                 negative_slope=.2,
                 residual=False,
                 DNN_layers1=2,
                 DNN_layers2=1):
        super(DCC_GNN, self).__init__()
        self.gat_net = GAT(tokens_size, in_dim, GNN_hidden_dims, graph_dim, num_heads,
                           GNN_layers, feat_drop, attn_drop, negative_slope, residual)
        self.dnn = DNN(graph_dim, graph_dim, DNN_hidden_dims1, DNN_hidden_dims2, class_num,
                           DNN_layers1, DNN_layers2)

    def forward(self, inputs):
        input1, input2 = inputs
        vec1 = self.gat_net(input1)
        vec2 = self.gat_net(input2)
        out = self.dnn((vec1, vec2))
        return out


class DCC_GNN_cosine(nn.Module):
    def __init__(self,
                 tokens_size,
                 in_dim,
                 GNN_hidden_dims,
                 graph_dim,
                 num_heads,
                 GNN_layers=4,
                 feat_drop=.2,
                 attn_drop=.2,
                 negative_slope=.2,
                 residual=False):
        super(DCC_GNN_cosine, self).__init__()
        self.gat_net = GAT(tokens_size, in_dim, GNN_hidden_dims, graph_dim, num_heads,
                           GNN_layers, feat_drop, attn_drop, negative_slope, residual)

    def forward(self, inputs):
        input1, input2 = inputs
        vec1 = self.gat_net(input1)
        vec2 = self.gat_net(input2)
        out = F.cosine_similarity(vec1, vec2)
        return out
