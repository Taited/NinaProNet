from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import os
import re


class NinaProDataset(Dataset):
    def __init__(self, root=None, random_sample=1024, window_length=150, overlap=0.6, transform=None, butterWn=None):
        super(NinaProDataset)
        if not os.path.exists(root):
            raise RuntimeError('The file path did not exist.')
        self.root = root
        self.window_length = window_length
        self.overlap = overlap
        self.transform = transform
        self.data = None
        self.label = None

        sub_file_names = os.listdir(self.root)
        for name in sub_file_names:
            sample_name = re.findall(r"_(s\d+)", name)[0]
            path = os.path.join(self.root, name, name, sample_name+'_A1_E1.mat')
            matlab_variable_dict = loadmat(path)
            if self.data is None:
                self.data = matlab_variable_dict['emg']
                self.label = matlab_variable_dict['restimulus']
            else:
                self.data = np.vstack((self.data, matlab_variable_dict['emg']))
                self.label = np.vstack((self.label, matlab_variable_dict['restimulus']))

        # 巴特沃斯滤波处理
        if butterWn is not None:
            b, a = signal.butter(N=2, Wn=butterWn)
            self.data = signal.filtfilt(b, a, self.data, axis=0)

        self.class_num = np.max(self.label) + 1  # take label 0 into account
        # segment the signals from label
        self.parsed_label = self.parse_label()
        self.min_signal, self.max_signal = self.min_max_signal()
        self.length = random_sample

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if item >= self.__len__():
            raise IndexError

        # # set img offset
        label_id = random.randint(0, self.class_num - 1)
        label_seg_id = random.randint(0, len(self.parsed_label[label_id]) - 1)
        seg_begin, seg_end = self.parsed_label[label_id][label_seg_id]
        label = np.zeros(1, dtype=float)
        label[0] = self.label[seg_begin, 0].copy()
        # label = self.onehot(label)
        data = self.data[seg_begin:seg_end, :].copy()  #直接通过切片得到的数据是不连续的，不通过copy一下转换成tensor时会报错

        sample = {'data': data, 'label': label}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def parse_label(self):
        parsed_label = [[] for i in range(self.class_num)]  # 初始化一个长度为class_num的二维列表
        length = self.window_length
        step = int(length * (1 - self.overlap))
        begin = 0
        end = length
        while end < self.label.shape[0]:
            segment = self.label[begin:end, 0]
            # 说明该段不具备label的重叠，载入数据中
            if len(np.unique(segment)) == 1:
                label_id = self.label[begin, 0]
                parsed_label[label_id].append([begin, end])
            begin += step
            end += step
        return parsed_label

    def min_max_signal(self):
        min_signal = np.min(self.data)
        max_signal = np.max(self.data)
        return min_signal, max_signal

    def onehot(self, label_id):
        label = np.zeros(self.class_num, dtype=float)
        label[label_id] = 1
        return label


class ToTensor(object):
    def __call__(self, sample):
        sample['data'], sample['label'] = torch.Tensor(sample['data']), torch.LongTensor(sample['label'])
        sample['data'] = sample['data'].transpose(0, 1)
        # sample['data'] = torch.unsqueeze(sample['data'], dim=0)
        sample['label'] = torch.unsqueeze(sample['label'], dim=0)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        sample['data'] = F.interpolate(sample['data'], self.size)
        return sample


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        data = sample['data']
        for i in range(3):
            data[:, i, :, :] = (data[:, i, :, :] - self.mean[i]) / self.std[i]
        sample['data'] = data
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


if __name__ == '__main__':
    root = r'D:\Dataset\NinaproEMG\DB1'
    cutoff_frequency = 45
    sampling_frequency = 100
    wn = 2 * cutoff_frequency / sampling_frequency
    myDataset = NinaProDataset(root=root, butterWn=wn)
    label_sampled = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    length_list = []
    for batch_id, sample_batch in enumerate(myDataset):
        data, label = sample_batch['data'], sample_batch['label']
        # print(batch_id)
        label_sampled[int(np.where(label == 1)[0])] += 1
        length_list.append(data.shape[0])
    print(label_sampled)
    print(np.mean(np.array(length_list)))
    plt.bar(x=np.arange(0, 13), height=np.array(label_sampled))
    plt.show()
