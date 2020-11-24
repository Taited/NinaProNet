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
    def __init__(self, root=None, transform=None, butterWn=None):
        super(NinaProDataset)
        if not os.path.exists(root):
            raise RuntimeError('The file path did not exist.')
        self.root = root
        self.data = []
        self.label = []
        self.transform = transform
        self.patient_num = 0
        sub_file_names = os.listdir(self.root)
        for name in sub_file_names:
            sample_name = re.findall(r"_(s\d+)", name)[0]
            path = os.path.join(self.root, name, name, sample_name+'_A1_E1.mat')
            matlab_variable_dict = loadmat(path)
            emg = matlab_variable_dict['emg']
            if butterWn is not None:
                b, a = signal.butter(2, butterWn)
                # plt.plot(emg[400:1000, 3])
                emg = signal.filtfilt(b, a, emg, axis=0)
                # plt.plot(emg[400:1000, 3])
                # plt.show()
                # stop = 1
            self.data.append(emg)
            self.label.append(matlab_variable_dict['restimulus'])
            self.patient_num += 1
        self.class_num = np.max(self.label[0]) + 1  # take label 0 into account
        # segment the signals from label
        self.parsed_label, self.length = self.__parse_label()
        self.min, self.max = self.__min_max()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if item >= self.__len__():
            raise IndexError

        # # set img offset
        patient_id = random.randint(0, self.patient_num - 1)
        label_seg_id = random.randint(0, len(self.parsed_label[patient_id]) - 1)
        label = self.parsed_label[patient_id][label_seg_id]
        data = self.data[patient_id][label[0]:label[1], :]
        label = self.onehot(self.label[patient_id][label[0]])
        sample = {'data': data, 'label': label}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __parse_label(self):
        parsed_label = []
        length = 0
        patient_id = 1
        for label in self.label:
            flag = 0
            label_seg = [[0]]
            label_id = 0
            for i in range(label.shape[0]):
                if flag == label[i]:
                    continue
                label_seg[label_id].append(i - 1)
                label_id += 1
                label_seg.append([i])
                flag = label[i]
                length += 1
            label_seg[label_id].append(i)
            parsed_label.append(label_seg)
            print('Label of Patient id {} parsed'.format(patient_id))
            patient_id += 1
        return parsed_label, length

    def __min_max(self):
        _min = 10
        _max = 0
        for data in self.data:
            if np.min(data) < _min:
                _min = np.min(data)
            if np.max(data) > _max:
                _max = np.max(data)
        return _min, _max

    def onehot(self, label_id):
        label = np.zeros(self.class_num, dtype=float)
        label[label_id] = 1
        return label


class ToTensor(object):
    def __call__(self, sample):
        sample['data'], sample['label'] = torch.Tensor(sample['data']), torch.Tensor(sample['label'])
        sample['data'] = torch.unsqueeze(sample['data'], dim=0)
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
    root = r'E:\Datasets\NinaproEMG'
    cutoff_frequency = 45
    sampling_frequency = 100
    wn = 2 * cutoff_frequency / sampling_frequency
    myDataset = NinaProDataset(root=root, butterWn=wn)
    label_sampled = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    length_list = []
    for batch_id, sample_batch in enumerate(myDataset):
        data, label = sample_batch['data'], sample_batch['label']
        print(batch_id)
        label_sampled[int(np.nonzero(label)[0])] += 1
        length_list.append(data.shape[0])
    print(label_sampled)
    print(np.mean(np.array(length_list)))
    plt.bar(x=np.arange(0, 13), height=np.array(label_sampled))
    plt.show()
