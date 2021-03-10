import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self,
                 in_dim1,
                 in_dim2,
                 DNN_hidden_dims1,
                 DNN_hidden_dims2,
                 class_num=2,
                 DNN_layers1=2,
                 DNN_layers2=1):
        super(DNN, self).__init__()
        self.in_dim = in_dim1 + in_dim2
        self.dnn1_layers = DNN_layers1
        self.dnn1 = nn.Sequential()
        if DNN_layers1 > 0:
            self.dnn1.add_module('fc1_{}'.format(1), nn.Linear(self.in_dim, DNN_hidden_dims1[0]))
            self.dnn1.add_module('relu1_{}'.format(1), nn.ReLU())
            if DNN_layers1 > 1:
                for i in range(1, DNN_layers1):
                    self.dnn1.add_module('fc1_{}'.format(i + 1), nn.Linear(DNN_hidden_dims1[i - 1], DNN_hidden_dims1[i]))
                    self.dnn1.add_module('relu1_{}'.format(i + 1), nn.ReLU())
        self.dnn2 = nn.Sequential()
        dnn1_out_dim = DNN_hidden_dims1[-1] if DNN_layers1 > 0 else self.in_dim
        if DNN_layers2 > 1:
            self.dnn2.add_module('fc2_{}'.format(1), nn.Linear(dnn1_out_dim, DNN_hidden_dims2[0]))
            self.dnn2.add_module('relu2_{}'.format(1), nn.ReLU())
            if DNN_layers2 > 2:
                for i in range(1, DNN_layers2 - 1):
                    self.dnn1.add_module('fc2_{}'.format(i + 1),
                                         nn.Linear(DNN_hidden_dims2[i - 1], DNN_hidden_dims2[i]))
                    self.dnn2.add_module('relu2_{}'.format(i + 1), nn.ReLU())
            self.dnn2.add_module('fc2_{}'.format(DNN_layers2), nn.Linear(DNN_hidden_dims2[-1], class_num))
        else:
            self.dnn2.add_module('fc2_{}'.format(1), nn.Linear(dnn1_out_dim, class_num))

    def forward(self, inputs):
        x1, x2 = inputs
        x12 = torch.cat((x1, x2), 1)
        x21 = torch.cat((x2, x1), 1)
        if self.dnn1_layers > 0:
            x12 = self.dnn1(x12)
            x21 = self.dnn1(x21)
        h = (x12 + x21) / 2
        out = self.dnn2(h)
        return out
