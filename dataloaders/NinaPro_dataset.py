from torch.utils.data import Dataset
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import random


class NinaPro_dataset(Dataset):
    def __init__(self, root=None):
        super(NinaPro_dataset)
        if not os.path.exists(root):
            raise RuntimeError('The file path did not exist.')
        self.root = root
        self.data = []
        self.label = []
        self.patient_num = 0
        sub_file_names = os.listdir(self.root)
        for name in sub_file_names:
            sample_name = re.findall(r"_(s\d+)", name)[0]
            path = os.path.join(self.root, name, name, sample_name+'_A1_E1.mat')
            matlab_variable_dict = loadmat(path)
            self.data.append(matlab_variable_dict['emg'])
            self.label.append(matlab_variable_dict['restimulus'])
            self.patient_num += 1
        # segment the signals from label
        self.parsed_label, self.length = self.__parse_label()
        self.min, self.max = self.__min_max()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if item >= self.__len__():
            raise IndexError
        # set img offset
        patient_id = random.randint(0, self.patient_num - 1)
        label_seg_id = random.randint(0, len(self.parsed_label[patient_id]) - 1)
        label = self.parsed_label[patient_id][label_seg_id]
        data = self.data[patient_id][label[0]:label[1], :]
        return data, self.label[patient_id][label[0]]

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


if __name__ == '__main__':
    root = r'D:\Dataset\NinaproEMG'
    myDataset = NinaPro_dataset(root=root)
    label_sampled = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    length_list = []
    for batch_id, (data, label) in enumerate(myDataset):
        print(batch_id)
        label_sampled[int(label)] += 1
        length_list.append(data.shape[0])
    print(label_sampled)
    print(np.mean(np.array(length_list)))
    plt.bar(x=np.arange(0, 13), height=np.array(label_sampled))
    plt.show()
