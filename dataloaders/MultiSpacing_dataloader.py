import os
import torch
import numpy as np
import numbers
from glob import glob
from torch.utils.data import Dataset
import itertools
from torch.utils.data.sampler import Sampler
from data_process.data_process_func import *

class MultiSpacingDataloader(Dataset):
    """ Adapt Dataset """
    def __init__(self, config=None, split='train', num=None, transform=None, random_sample=True):
        self._data_root = config['data_root']
        self._hr_filename = config['hr_name']
        self._mr_filename = config['mr_name']
        self._lr_filename = config['lr_name']
        self._label_hr_filename = config['label_hr_name']
        self._label_mr_filename = config['label_mr_name']
        self._label_lr_filename = config['label_lr_name']
        self.patch_ratio_dic = config['patch_ratio_dic']
        self.filename_dic = {}
        for filename in ['hr','mr','lr','label_hr', 'label_mr', 'label_lr']:
            self.filename_dic[filename]=config['{0:}_name'.format(filename)]
        self._coarseg_filename = config.get('coarseg_name', False)
        self._distance_filename = config.get('dis_name', False)
        self._iternum = config['iter_num']
        self.split = split
        self.transform = transform
        self.sample_list = []
        self.random_sample = random_sample
        if split in ['train', 'valid', 'test']:
            self.image_list = os.listdir(os.path.join(self._data_root, split))
        else:
            ValueError('please input choose correhr mode! i.e."train" "valid" "test"')
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        if self.random_sample == True:
            return self._iternum
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        if self.random_sample == True:
            image_fold = random.sample(self.image_list, 1)[0]
        else:
            image_fold = self.image_list[idx]
        patient_path = os.path.join(self._data_root, self.split, image_fold)
        sample = {'patient_path':patient_path}
        for file_name in self.filename_dic.keys():
            path = os.path.join(patient_path, self.filename_dic[file_name])
            sample[file_name]=load_nifty_volume_as_array(path)
        if self._coarseg_filename:
            coarseg_path = os.path.join(patient_path, self._coarseg_filename)
            coarseg = load_nifty_volume_as_array(coarseg_path)
            sample['coarseg']= coarseg
        if self._distance_filename:
            distance_path = os.path.join(patient_path, self._distance_filename)
            distance = load_nifty_volume_as_array(distance_path)
            sample['distance']=distance
        if self.transform:
            sample = self.transform(sample)
        return sample


class CenterCrop(MultiSpacingDataloader):
    def __init__(self, output_size):
        super(MultiSpacingDataloader, self).__init__()
        self.output_size = output_size

    def __call__(self, sample):
        hr, mr, lr, label, patient_path= sample['hr'], sample['mr'],sample['lr'],sample['label'],sample['patient_path']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            hr = np.pad(hr, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            mr = np.pad(mr, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            lr = np.pad(lr, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = hr.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        hr = hr[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        mr = mr[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        lr = lr[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'hr': hr, 'label': label, 'mr':mr, 'lr':lr, 'patient_path': patient_path}

class CropBound(MultiSpacingDataloader):
    def __init__(self, pad=[0,0,0], mode='label'):
        super(MultiSpacingDataloader, self).__init__()
        self.pad = pad
        self.mode = mode
    def __call__(self, sample):
        hr, mr, lr, label, patient_path = sample['hr'],sample['mr'],sample['lr'], sample['label'],sample['patient_path']
        file = sample[self.mode]
        file_size = file.shape
        nonzeropoint = np.asarray(np.nonzero(file))  # 得到非0点坐标,输出为一个3*n的array，3代表3个维度，n代表n个非0点在对应维度上的坐标
        maxpoint = np.max(nonzeropoint, 1).tolist()
        minpoint = np.min(nonzeropoint, 1).tolist()
        for i in range(len(self.pad)):
            maxpoint[i] = min(maxpoint[i] + self.pad[i], file_size[i])
            minpoint[i] = max(minpoint[i] - self.pad[i], 0)
        hr = hr[minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]]
        mr = mr[minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]]
        lr = lr[minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]]
        label = label[minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]]
        nsample = {'hr': hr, 'label': label, 'mr':mr, 'lr':lr, 'patient_path':patient_path}
        if 'coarseg' in sample:
            coarseg = sample['coarseg'][minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]]
            nsample['coarseg']=coarseg; nsample['crop_cor']=[minpoint, maxpoint]
        if 'distance' in sample:
            distance = sample['distance'][minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]]
            nsample['distance']=distance
        return nsample

class ExtractCertainClass(MultiSpacingDataloader):
    def __init__(self, class_wanted=[1]):
        super(MultiSpacingDataloader, self).__init__()
        self.class_wanted = class_wanted
    def __call__(self, sample):
        label = sample['label']
        nlabel = np.zeros_like(label)
        if 'coarseg' in sample:
            ncoarseg = np.zeros_like(sample['coarseg'])
        for i in range(len(self.class_wanted)):
            nlabel[np.where(label==self.class_wanted[i])]=i+1
            if 'coarseg' in sample:
                ncoarseg[np.where(sample['coarseg'] == self.class_wanted[i])] = i + 1
        sample['label']=nlabel
        if 'coarseg' in sample:
            sample['coarseg']=ncoarseg
        return sample

class RandomCrop(MultiSpacingDataloader):
    """
    Crop randomly the hr in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        super(MultiSpacingDataloader, self).__init__()
        self.output_size = output_size
        self.patch_ratio_dic = {'hr': 4, 'mr': 2, 'lr': 1, 'label_hr': 4, 'label_mr': 2, 'label_lr': 1}
    def __call__(self, sample):
        lr, patient_path = sample['lr'],sample['patient_path']

        # pad the sample if necessary
        if lr.shape[0] <= self.output_size[0] or lr.shape[1] <= self.output_size[1] or lr.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - lr.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - lr.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - lr.shape[2]) // 2 + 3, 0)
            for file_name in self.patch_ratio_dic.keys():
                sample[file_name] = np.pad(sample[file_name],[(self.patch_ratio_dic[file_name]*pw, self.patch_ratio_dic[file_name]*pw)
                    , (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if 'coarseg' in sample:
                sample['coarseg'] = np.pad(sample['coarseg'], [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = lr.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])
        for file_name in self.patch_ratio_dic.keys():
            sample[file_name] = sample[file_name][self.patch_ratio_dic[file_name]*w1:self.patch_ratio_dic[file_name]*(w1 + self.output_size[0]),
                                h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if 'coarseg' in sample:
            coarseg = sample['coarseg'][w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            sample['coarseg']=coarseg
        if 'distance' in sample:
            distance = sample['distance'][w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            sample['distance']=distance
        return sample

class RandomRotFlip(MultiSpacingDataloader):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        super(MultiSpacingDataloader, self).__init__()
        hr,mr,lr, label,patient_path = sample['hr'],sample['mr'],sample['lr'], sample['label'],sample['patient_path']
        k = np.random.randint(0, 4)
        hr = np.rot90(hr, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        hr = np.flip(hr, axis=axis).copy()
        mr = np.flip(mr, axis=axis).copy()
        lr = np.flip(lr, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        if 'coarseg' in sample:
            coarseg = np.rot90(sample['coarseg'], k)
            coarseg = np.flip(coarseg, axis=axis).copy()
            return {'hr': hr, 'mr':mr, 'lr':lr,'label': label, 'coarseg':coarseg, 'patient_path':patient_path}
        else:
            return {'hr': hr, 'mr':mr, 'lr':lr,'label': label}


class RandomNoise(MultiSpacingDataloader):
    def __init__(self, mu=0, sigma=0.1):
        super(MultiSpacingDataloader, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        hr,mr,lr, label,patient_path = sample['hr'],sample['mr'],sample['lr'], sample['label'],sample['patient_path']
        noise = np.clip(self.sigma * np.random.randn(hr.shape[0], hr.shape[1], hr.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        hr = hr + noise
        mr = mr + noise
        lr = lr + noise
        if 'coarseg' in sample:
            return {'hr': hr, 'mr':mr, 'lr':lr,'label': label, 'coarseg':sample['coarseg'],'patient_path':sample['patient_path']}
        else:
            return {'hr': hr, 'mr':mr, 'lr':lr,'label': label,'patient_path':sample['patient_path']}

class RandomRotate(MultiSpacingDataloader):
    def __init__(self, p=0.5, axes=(0,1), max_degree=0):
        super(MultiSpacingDataloader, self).__init__()
        self.p = p
        self.axes = axes
        self.max_degree = max_degree


    def __call__(self, sample):
        hr,mr,lr, label,patient_path = sample['hr'],sample['mr'],sample['lr'], sample['label'],sample['patient_path']
        if random.random() < self.p:
            if isinstance(self.max_degree, numbers.Number):
                if self.max_degree < 0:
                    raise ValueError("If degrees is a single number, it must be positive.")
                degrees = (-self.max_degree, self.max_degree)
            else:
                if len(self.max_degree) != 2:
                    raise ValueError("If degrees is a sequence, it must be of len 2.")
                degrees = self.max_degree
            if len(self.axes) != 2:
                axes = random.sample(self.axes, 2)
            else:
                axes = self.axes
            angle = random.uniform(degrees[0], degrees[1])
            hr = ndimage.rotate(hr, angle, axes=axes, order=0, reshape=False)
            mr = ndimage.rotate(mr, angle, axes=axes, order=0, reshape=False)
            lr = ndimage.rotate(lr, angle, axes=axes, order=0, reshape=False)
            label = ndimage.rotate(label, angle, axes=axes, order=0, reshape=False)
            if 'coarseg' in sample:
                coarseg = ndimage.rotate(sample['coarseg'], angle, axes=axes, order=0, reshape=False)
                return {'hr': hr, 'mr':mr, 'lr':lr,'label': label, 'coarseg':coarseg,'patient_path':patient_path}
            else:
                return {'hr': hr, 'mr':mr, 'lr':lr,'label': label,'patient_path':patient_path}
        else:
            return sample

class RandomScale(MultiSpacingDataloader):
    def __init__(self, p=0.5, axes=(0,1), max_scale=1):
        super(MultiSpacingDataloader, self).__init__()
        self.p = p
        self.axes = axes
        self.max_scale = max_scale


    def __call__(self, sample):
        hr,mr,lr, label,patient_path = sample['hr'],sample['mr'],sample['lr'], sample['label'],sample['patient_path']
        if random.random() < self.p:
            if isinstance(self.max_scale, numbers.Number):
                if self.max_scale < 0:
                    raise ValueError("If degrees is a single number, it must be positive.")
                scale = (1/self.max_scale, self.max_scale)
            else:
                if len(self.max_scale) != 2:
                    raise ValueError("If degrees is a sequence, it must be of len 2.")
                scale = self.max_scale
            scale = random.uniform(scale[0], scale[1])
            hr = ndimage.zoom(hr, scale,  order=0)
            mr = ndimage.zoom(mr, scale, order=0)
            lr = ndimage.zoom(lr, scale, order=0)
            label = ndimage.zoom(label, scale,  order=0)
            if 'coarseg' in sample:
                coarseg = ndimage.rotate(sample['coarseg'], scale, order=0)
                return {'hr': hr, 'mr':mr, 'lr':lr, 'label': label, 'coarseg': coarseg,'patient_path':patient_path}
            else:
                return {'hr': hr, 'mr':mr, 'lr':lr, 'label': label,'patient_path':patient_path}
        else:
            return sample

class CreateOnehotLabel(MultiSpacingDataloader):
    def __init__(self, num_classes):
        super(MultiSpacingDataloader, self).__init__()
        self.num_classes = num_classes

    def __call__(self, sample):
        label = sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        sample['onehot_label']=onehot_label
        return sample


class ToTensor(MultiSpacingDataloader):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, concat_coarseg=False):
        super(MultiSpacingDataloader, self).__init__()
        self.concat_coarseg=concat_coarseg
    def __call__(self, sample):
        for file in ['hr','mr','lr']:
            sample[file] = torch.from_numpy(sample[file]).unsqueeze(dim=0).float()
        for file in ['label_hr','label_mr','label_lr']:
            sample[file] = torch.from_numpy(sample[file]).long()

        if 'onehot_label' in sample:
            sample['onehot_label']=torch.from_numpy(sample['onehot_label']).long()
        if 'coarseg' in sample:
            coarseg = torch.from_numpy(sample['coarseg']).unsqueeze(dim=0).float()
            sample['coarseg']=coarseg
        return sample

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collehr data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
