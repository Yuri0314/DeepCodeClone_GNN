import os

import torch

from torch.utils.data import Dataset
from torch.utils.data import random_split
from tqdm import tqdm


class ClonePairDataset(Dataset):
    def __init__(self, file2graph, file2tokenIdx, clone_pairs, desc):
        self.data = []
        for clone_pair in tqdm(clone_pairs, desc=desc):
            tmp_lst = clone_pair.split()
            idx_list1 = file2tokenIdx[tmp_lst[0]]
            edges1, edge_types1 = file2graph[tmp_lst[0]]
            idx_list2 = file2tokenIdx[tmp_lst[1]]
            edges2, edge_types2 = file2graph[tmp_lst[1]]

            self.data.append(([idx_list1, edges1, edge_types1], [idx_list2, edges2, edge_types2], int(tmp_lst[2])))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def generate_model_input(file2graph, file2tokenIdx, dataset='googlejam4_src', validation_split=.2):
    with open(os.path.join('.', dataset + '_true_clone_pairs.dat'), 'r') as f:
        true_pairs = f.readlines()
    with open(os.path.join('.', dataset + '_false_clone_pairs.dat'), 'r') as f:
        false_pairs = f.readlines()

    true_dataset = ClonePairDataset(file2graph, file2tokenIdx, true_pairs,
                                    'True clone pair data generating')
    false_dataset = ClonePairDataset(file2graph, file2tokenIdx, false_pairs,
                                     'False clone pair data generating')

    # 划分数据集
    true_test_size = int(validation_split * len(true_dataset))
    true_train_size = len(true_dataset) - true_test_size
    true_train, true_test = random_split(true_dataset, [true_train_size, true_test_size], \
                                         generator=torch.Generator().manual_seed(len(true_dataset)))
    false_test_size = int(validation_split * len(false_dataset))
    false_train_size = len(false_dataset) - false_test_size
    false_train, false_test = random_split(false_dataset, [false_train_size, false_test_size], \
                                           generator=torch.Generator().manual_seed(len(false_dataset)))

    train_data = true_train + false_train
    test_data = true_test + false_test

    return train_data, test_data