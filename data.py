import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


class CustData(Dataset):

    def __init__(self, x, y=0, y_max=0, train=False):

        self.x = x
        self.y = y
        self.y_max = y_max
        self.idx_list = x.index
        self.train = train

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        data = torch.FloatTensor(self.x.iloc[idx, :].values)
        if self.train:
            label_ = self.y[idx]
            label = torch.cat((torch.ones(label_),
                               torch.zeros(self.y_max-label_)))
            samples = {'data': data, 'label': label}
        else:
            samples = {'data': data}
        return samples
