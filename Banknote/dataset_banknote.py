# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = dataset_banknote
@author = wly
@create_time = 2022/9/24 16:37
"""
import torch
from torch.utils.data import DataLoader
from config import HP
import numpy as np


class BanknoteDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.dataset = np.loadtxt(data_path, delimiter=',')

    def __getitem__(self, idx):
        item = self.dataset[idx]
        x, y = item[:HP.in_features], item[HP.in_features:]
        return torch.Tensor(x).float().to(HP.device), torch.Tensor(y).squeeze().long().to(HP.device)

    def __len__(self):
        return self.dataset.shape[0]


'''
if __name__ == '__main__':
    bkdataset = BanknoteDataset(HP.testset_path)
    bkdataloader = DataLoader(bkdataset, batch_size=16, shuffle=True, drop_last=True)

    for batch in bkdataloader:
        x, y = batch
        print(x)
        print(y)
        break
'''
