from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt


class RNN_sin_dataset(Dataset):
    def __init__(self, is_train=True, length=100):
        self.data = 1
        self.is_train = is_train
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        data, label, seq_len = self.data[item]
        return data
