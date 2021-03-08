import torch

from model.model import DCC_GNN

model = DCC_GNN(6034, 128, [128, 128, 128, 128], 128, [3, 3, 3, 3],
                [128, 64], [32], DNN_layers1=2, DNN_layers2=2)
print(model)
# x1 = torch.randn(1, 128)
# x2 = torch.randn(1, 128)
# out = model([x1, x2])
# print(out)

# import torch
#
# from model.gat_net import GAT
#
# model = GAT(3000, 128, [128, 128, 128, 128], 128, [3, 3, 3, 3])
# print(model)
#


aa = {1, 2, 3}
print(aa)
aa.update({2, 3, 4, 5})
print(aa)