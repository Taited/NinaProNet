import os
import torch
import numpy as np
import numbers
from scipy import ndimage
from glob import glob
from torch.utils.data import Dataset
import random
import h5py
import itertools
from torch.utils.data.sampler import Sampler
from data_process.data_process_func import *


class StrusegDataloader(Dataset):
    """ structseg Dataset """
    def __init__(self, config=None, split='train', num=None, transform=None, random_sample=True):
        self._data_root = config['data_root']
        self._image_filename = config['image_name']
        self._label_filename = config['label_name']
        self._dis
        self._coarseg_filename = config.get('coarseg_name', None)
        self._iternum = config['iter_num']
        self.split = split
        self.transform = transform
        self.sample_list = []
        self.random_sample = random_sample
        if split in ['train', 'valid', 'test']:
            self.image_list = os.listdir(self._data_root + '/' + split)
        else:
            ValueError('please input choose correct mode! i.e."train" "valid" "test"')
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
        image_path = os.path.join(patient_path, self._image_filename)
        label_path = os.path.join(patient_path, self._label_filename)
        image = load_nifty_volume_as_array(image_path)
        label = load_nifty_volume_as_array(label_path)
        if self._coarseg_filename:
            coarseg_path = os.path.join(patient_path, self._coarseg_filename)
            coarseg = load_nifty_volume_as_array(coarseg_path)
            sample = {'image': image, 'label': label, 'coarseg': coarseg, 'patient_path': patient_path}
            if self.transform:
                sample = self.transform(sample)
            return sample
        else:
            sample = {'image': image, 'label': label, 'patient_path': patient_path}
            if self.transform:
                sample = self.transform(sample)
            return sample


class PositionDataloader(Dataset):
    """ structseg Dataset position """
    def __init__(self, config=None, split='train', num=None, transform=None):
        self._data_root = config['data_root']
        self._image_filename = config['image_name']
        self._iternum = config['iter_num']
        self.split = split
        self.transform = transform
        self.sample_list = []
        if split=='train':
            self.image_list = os.listdir(self._data_root+'/'+'train')
        elif split == 'valid':
            self.image_list = os.listdir(self._data_root + '/' + 'valid')
        elif split == 'test':
            self.image_list = os.listdir(self._data_root + '/' + 'test')
        else:
            ValueError('please input choose correct mode! i.e."train" "valid" "test"')
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        if self.split == 'train':
            return self._iternum
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        if self.split == 'train':
            image_fold = random.sample(self.image_list, 1)[0]
        else:
            image_fold = self.image_list[idx]
        image_path = os.path.join(self._data_root, self.split, image_fold, self._image_filename)
        image, spacing = load_nifty_volume_as_array(image_path, return_spacing=True)
        spacing = np.asarray([3,1,1])
        sample = {'image': image, 'spacing':spacing}
        if self.transform:
            sample = self.transform(sample)

        return sample


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class CropBound(object):
    def __init__(self, pad=[0,0,0], mode='label'):
        self.pad = pad
        self.mode = mode
    def __call__(self, sample):
        image,label = sample['image'], sample['label']
        file = sample[self.mode]
        file_size = file.shape
        nonzeropoint = np.asarray(np.nonzero(file))  # 得到非0点坐标,输出为一个3*n的array，3代表3个维度，n代表n个非0点在对应维度上的坐标
        maxpoint = np.max(nonzeropoint, 1).tolist()
        minpoint = np.min(nonzeropoint, 1).tolist()
        for i in range(len(self.pad)):
            maxpoint[i] = min(maxpoint[i] + self.pad[i], file_size[i])
            minpoint[i] = max(minpoint[i] - self.pad[i], 0)
        image = image[minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]]
        label = label[minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]]
        if 'coarseg' in sample:
            coarseg = sample['coarseg'][minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]]
            return {'image': image, 'label': label, 'coarseg': coarseg, 'crop_cor':[minpoint, maxpoint]}
        else:
            return {'image': image, 'label': label}


class ExtractCertainClass(object):
    def __init__(self, class_wanted=[1]):
        self.class_wanted = class_wanted
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        nlabel = np.zeros_like(label)
        if 'coarseg' in sample:
            ncoarseg = np.zeros_like(sample['coarseg'])
        for i in range(len(self.class_wanted)):
            nlabel[np.where(label==self.class_wanted[i])]=i+1
            if 'coarseg' in sample:
                ncoarseg[np.where(sample['coarseg'] == self.class_wanted[i])] = i + 1
        if 'coarseg' in sample:
            return {'image': image, 'label': nlabel, 'coarseg': ncoarseg}
        else:
            return {'image': image, 'label':nlabel}

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if 'coarseg' in sample:
                sample['coarseg'] = np.pad(sample['coarseg'], [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if 'coarseg' in sample:
            coarseg = sample['coarseg'][w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'coarseg': coarseg}
        else:
            return {'image': image, 'label': label}

class RandomPositionSeveralCrop(object):
    """
    Crop several randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, crop_num):
        self.output_size = output_size
        self.crop_num = crop_num
    def __call__(self, sample):
        image= sample['image']
        n_sample = {}
        # pad the sample if necessary
        if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        for i in range(self.crop_num):
            w1 = np.random.randint(0, w - self.output_size[0])
            h1 = np.random.randint(0, h - self.output_size[1])
            d1 = np.random.randint(0, d - self.output_size[2])

            label = w1+self.output_size//2
            n_image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            n_sample['image{}'.format(i)]=n_image
            n_sample['label{}'.format(i)]=label
        return n_sample

class RandomPositionDoubleCrop(object):
    """
    Crop double randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, sample):
        image,spacing= sample['image'],sample['spacing']
        n_sample = {}
        # pad the sample if necessary
        if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        relative_position = np.zeros(6)
        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])
        w2 = np.random.randint(0, w - self.output_size[0])
        h2= np.random.randint(0, h - self.output_size[1])
        d2 = np.random.randint(0, d - self.output_size[2])
        if  w1>w2:
            relative_position[0] = 1 
        else :
            relative_position[1]=1
        if  h1>h2:
            relative_position[2] = 1 
        else :
            relative_position[3]=1
        if  d1>d2:
            relative_position[4] = 1 
        else :
            relative_position[5]=1
        label1 = np.asarray([w1,h1,d1])
        label2 = np.asarray([w2,h2,d2])
        image1 = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image2 = image[w2:w2 + self.output_size[0], h2:h2 + self.output_size[1], d2:d2 + self.output_size[2]]
        n_sample['image1']=image1
        n_sample['image2']=image2
        n_sample['label1']=label1
        n_sample['label2']=label2
        spacing = np.asarray([3,1,1])
        n_sample['distance']=(label1-label2)*spacing
        n_sample['rela_poi']=relative_position
        return n_sample

class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        if 'coarseg' in sample:
            coarseg = np.rot90(sample['coarseg'], k)
            coarseg = np.flip(coarseg, axis=axis).copy()
            return {'image': image, 'label': label, 'coarseg':coarseg}
        else:
            return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        if 'coarseg' in sample:
            return {'image': image, 'label': label, 'coarseg':sample['coarseg']}
        else:
            return {'image': image, 'label': label}

class RandomRotate(object):
    def __init__(self, p=0.5, axes=(0,1), max_degree=0):
        self.p = p
        self.axes = axes
        self.max_degree = max_degree


    def __call__(self, sample):
        image, label = sample['image'], sample['label']
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
            image = ndimage.rotate(image, angle, axes=axes, order=0, reshape=False)
            label = ndimage.rotate(label, angle, axes=axes, order=0, reshape=False)
            if 'coarseg' in sample:
                coarseg = ndimage.rotate(sample['coarseg'], angle, axes=axes, order=0, reshape=False)
                return {'image': image, 'label': label, 'coarseg': coarseg}
            else:
                return {'image': image, 'label': label}
        else:
            return sample

class RandomScale(object):
    def __init__(self, p=0.5, axes=(0,1), max_scale=1):
        self.p = p
        self.axes = axes
        self.max_scale = max_scale


    def __call__(self, sample):
        image, label = sample['image'], sample['label']
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
            image = ndimage.zoom(image, scale,  order=0)
            label = ndimage.zoom(label, scale,  order=0)
            if 'coarseg' in sample:
                coarseg = ndimage.rotate(sample['coarseg'], scale, order=0)
                return {'image': image, 'label': label, 'coarseg': coarseg}
            else:
                return {'image': image, 'label': label}
        else:
            return sample

class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, doubleinput=False):
        self.doubleinput=doubleinput
    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            if 'coarseg' in sample:
                coarseg = sample['coarseg']
                coarseg = coarseg.reshape(1, coarseg.shape[0], coarseg.shape[1], coarseg.shape[2]).astype(np.float32)
                coarseg = torch.from_numpy(coarseg)
                image = torch.from_numpy(image)
                if self.doubleinput:
                    return {'image': torch.cat((image,coarseg.float()), 0), 'label': torch.from_numpy(sample['label']).long(), 'coarseg':coarseg.long()}
                else:
                    return {'image': image, 'label': torch.from_numpy(sample['label']).long(), 'coarseg':coarseg.long()}
            else:
                return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}

class ToPositionTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image1 = sample['image1']
        image2 = sample['image2']
        image1 = image1.reshape(1, image1.shape[0], image1.shape[1], image1.shape[2]).astype(np.float32)
        image2 = image2.reshape(1, image2.shape[0], image2.shape[1], image2.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image1': torch.from_numpy(image1), 'image2': torch.from_numpy(image2),'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image1': torch.from_numpy(image1),'image2': torch.from_numpy(image2), 
            'distance': torch.from_numpy(sample['distance']).float(), 'rela_poi':torch.from_numpy(sample['rela_poi']).float()}

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
