from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import numpy as np
import random
import torch
import os
import re


class EMGDataset(Dataset):
    def __init__(self, root=None, split='train', random_sample=1024, window_length=128, overlap=0.6,
                 transform=None, is_filter=False):
        super(EMGDataset)
        # load parameters
        self.root = os.path.join(root, split)
        self.window_length = window_length
        self.overlap = overlap
        self.split = split
        self.transform = transform
        self.data = None
        self.label = None
        self.class_num = 17  # since take label 0 into account, beginning with 1

        # get raw label
        # split by train or valid
        patients_names = os.listdir(self.root)
        # traverse patients
        for patients in patients_names:
            sub_file_names = os.listdir(os.path.join(self.root, patients))
            label = None
            data = None
            for name in sub_file_names:
                print('\tProcessing entity ' + name)
                experiment_type = re.findall(r'E(\d+).', name)[0]
                base_class_num = 4 * (int(experiment_type) - 1)
                path = os.path.join(self.root, patients, name)
                matlab_variable_dict = loadmat(path)
                raw_label = matlab_variable_dict['label']
                raw_data = matlab_variable_dict['emg']
                raw_label[np.where(raw_label != 0)] += base_class_num
                if label is None:
                    label = raw_label
                    data = raw_data
                else:
                    label = np.vstack((label, raw_label))
                    data = np.vstack((data, raw_data))
            if self.label is None:
                self.label = {patients: label}
                self.data = {patients: data}
            else:
                self.label[patients] = label
                self.data[patients] = data
        # calculate max, min, mean, std
        my_dict = self.get_min_max_mean_std()
        self.max, self.min, self.mean, self.std = my_dict['max'], my_dict['min'], my_dict['mean'], my_dict['std']
        print('{} set mean: {}, std: {}'.format(split, self.mean, self.std))
        # segment the signals from label
        self.parsed_label = self.parse_label()

        # set sample iteration in train or valid
        if self.split == 'train':
            self.length = random_sample
        elif self.split == 'valid':  # 将所有的信号段拼接在一起，便于getitem按索引读入
            self.valid_label_seg = []
            self.valid_label = []
            for key in self.label:
                parsed_label = self.label[key]
                for i in range(len(parsed_label)):
                    for seg in parsed_label[i]:
                        self.valid_label_seg.append(seg)
                        self.valid_label.append(i)
            self.length = len(self.valid_label_seg)
        else:
            self.length = random_sample

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if item >= self.__len__():
            raise IndexError
        # 测试的时候顺序遍历所有的数据
        key_values = list(self.data.keys())
        key_id = random.randint(0, len(key_values) - 1)
        temp_data = self.data[key_values[key_id]]
        temp_label = self.label[key_values[key_id]]
        if self.split == 'valid':
            seg_begin, seg_end = self.valid_label_seg[item]
            label = np.zeros(1, dtype=float)
            label[0] = self.valid_label[item]
        else:
            # set img offset
            label_id = random.randint(0, self.class_num - 1)
            label_seg_id = random.randint(0, len(temp_label[label_id]) - 1)
            seg_begin, seg_end = temp_label[label_id][label_seg_id]
            label = np.zeros(1, dtype=float)
            label[0] = label_id

        data = temp_data[seg_begin:seg_end, :].copy()  # 直接通过切片得到的数据是不连续的，不通过copy一下转换成tensor时会报错
        sample = {'data': data, 'label': label, 'mean': self.mean, 'std': self.std}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def parse_label(self):
        if self.split == 'valid':
            self.overlap = 0
        for name in self.label:
            label = self.label[name]
            parsed_label = [[] for _ in range(self.class_num)]  # 初始化一个长度为class_num的二维列表
            length = self.window_length
            step = int(length * (1 - self.overlap))
            begin = 0
            end = length
            while end < label.shape[0]:
                segment = label[begin:end, 0]
                # 说明该段不具备label的重叠，载入数据中
                if len(np.unique(segment)) == 1:
                    label_id = label[begin, 0]
                    parsed_label[label_id].append([begin, end])
                begin += step
                end += step
            self.label[name] = parsed_label
        return parsed_label

    def get_min_max_mean_std(self):
        if self.data is None:
            raise Exception("There is no data!")
        min_list = []
        max_list = []
        num_sum = 0.0
        counter = 0
        for key in self.data:
            min_list.append(np.min(self.data[key], axis=0))
            max_list.append(np.max(self.data[key], axis=0))
            num_sum += np.sum(self.data[key], axis=0)
            counter += self.data[key].shape[0]
        num_mean = num_sum / counter
        num_std = 0.0
        for key in self.data:
            num_std += np.sum(np.square(self.data[key] - num_mean), axis=0)
        num_std = num_std / counter
        result = {'min': np.min(np.array(min_list), axis=0), 'max': np.max(np.array(max_list), axis=0),
                  'mean': num_mean, 'std': num_std}
        return result


class ToTensor(object):
    def __call__(self, sample):
        sample['data'], sample['label'] = torch.Tensor(sample['data']), torch.LongTensor(sample['label'])
        sample['data'] = sample['data'].transpose(0, 1)
        sample['label'] = torch.unsqueeze(sample['label'], dim=0)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToTensor2D(object):
    def __call__(self, sample):
        sample['data'], sample['label'] = torch.Tensor(sample['data']), torch.LongTensor(sample['label'])
        sample['data'] = sample['data'].transpose(0, 1)
        sample['data'] = sample['data'].unsqueeze(0)
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
    def __call__(self, sample):
        data = sample['data']
        mean = sample['mean']
        std = sample['std']
        for i in range(data.shape[1]):
            data[:, i] = (data[:, i] - mean[i]) / std[i]
        sample['data'] = data
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class FeatureExtractor(object):
    def __init__(self):
        # 获取类内的所有方法名字
        self.method_list = dir(self)

    def __call__(self, sample):
        feature_list = []
        # 由于继承了nn.module, 里面有一些方法不是计算特征的，要忽略掉
        for method in self.method_list:
            if method[0:2] == 'f_':
                func = getattr(self, method)
                feature_list.append(func(sample['data']))
        feature = np.array(feature_list)
        sample['data'] = feature.flatten()
        return sample

    @staticmethod
    def f_RMS(d):
        return np.sqrt(np.mean(np.square(d), axis=0))

    @staticmethod
    def f_MAV(d):
        return np.mean(np.abs(d), axis=0)

    @staticmethod  # 过零点次数
    def f_ZC(d):
        nZC = np.zeros(d.shape[1])
        th = np.mean(d, axis=0)
        th = np.abs(th)
        for i in range(1, d.shape[0]):
            for j in range(d.shape[1]):
                if d[i - 1, j] < th[j] < d[i, j]:
                    nZC[j] += 1
                elif d[i - 1, j] > th[j] > d[i, j]:
                    nZC[j] += 1
        return nZC / d.shape[0]

    @staticmethod  # slope sign change
    def f_SSC(d):
        nSSC = np.zeros(d.shape[1])
        th = np.mean(d, axis=0)
        th = np.abs(th)
        for i in range(2, d.shape[0]):
            diff1 = d[i] - d[i - 1]
            diff2 = d[i - 1] - d[i - 2]
            for j in range(d.shape[1]):
                if np.abs(diff1[j]) > th[j] and np.abs(diff2[j]) > th[j] and (diff1[j] * diff2[j]) < 0:
                    nSSC[j] += 1
        return nSSC / d.shape[0]

    @staticmethod
    def f_VAR(d):
        feature = np.var(d, axis=0)
        return feature


if __name__ == '__main__':
    root = r'D:\Dataset\SIA_delsys_16_movements_data'
    myDataset = EMGDataset(root=root, split='train', is_filter=False, transform=Normalize())
    label_sampled = np.linspace(1, myDataset.class_num, myDataset.class_num)
    length_list = []
    for i in range(len(myDataset.parsed_label)):
        length = len(myDataset.parsed_label[i])
        label_sampled[i] += length
    label_sampled /= np.sum(label_sampled)
    plt.subplot(2, 1, 1)
    plt.bar(x=np.arange(0, myDataset.class_num), height=label_sampled, label='original label number distribution')
    plt.gca().set(xlim=(0, myDataset.class_num), xlabel='label id', ylabel='ratio')
    plt.legend()
    label_sampled = np.linspace(1, myDataset.class_num, myDataset.class_num)

    # for i in range(len(myDataset)):
    #     sample_batch = myDataset[i]
    #     data, label = sample_batch['data'], sample_batch['label']
    #     label_sampled[int(label[0])] += 1
    #     length_list.append(data.shape[0])

    for sample_batch in myDataset:
        data, label = sample_batch['data'], sample_batch['label']
        label_sampled[int(label[0])] += 1
        length_list.append(data.shape[0])
    print(label_sampled)
    print(np.mean(np.array(length_list)))
    plt.subplot(2, 1, 2)
    label_sampled /= np.sum(label_sampled)
    plt.bar(x=np.arange(0, myDataset.class_num), height=label_sampled, label='sampleed label number distribution')
    plt.gca().set(xlim=(0, myDataset.class_num), ylim=(0.0, 0.10), xlabel='label id', ylabel='ratio')
    plt.legend()
    plt.tight_layout()
    plt.show()
