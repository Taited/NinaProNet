from __future__ import print_function
from networks.module import Module
import torch
import torch.nn as nn
from networks.layers import TriResSeparateConv3D
import torch.nn.functional as F
import numpy as np

class Adapt_transform(Module):
    """
    自适应Transform
    """
    def __init__(self, hu_lis=[0], norm_lis=[0], base_hu=-2000, base_norm=0):
        super(Adapt_transform, self).__init__()
        self.base_hu = base_hu
        self.base_norm = base_norm
        self.hu_lis = nn.Parameter(torch.FloatTensor(hu_lis))
        self.norm_lis = nn.Parameter(torch.FloatTensor(norm_lis))

    def forward(self,img):
        """
        :param img: tensor
        :param self.hu_lis:
        :param self.norm_lis:
        :param norm: norm or not
        :return:
        """
        for j in range(self.hu_lis.shape[0]): # 对所有transform类遍历
            cur_file = torch.zeros_like(img)
            for i in range(1, self.hu_lis.shape[1]): # 对一类中所有hu遍历
                hu_high = torch.sum(torch.abs(100*self.hu_lis[j,0:i+1]))
                hu_low = torch.sum(torch.abs(100*self.hu_lis[j,0:i]))
                norm_high = torch.sum(torch.abs(self.norm_lis[j,0:i+1]))
                norm_low = torch.sum(torch.abs(self.norm_lis[j,0:i]))
                mask = torch.where((img<(self.base_hu+hu_high))&(img>=(self.base_hu+hu_low)))
                k = (norm_high-norm_low)/(hu_high-hu_low)
                cur_file[mask] = k*(img[mask]-hu_low)+norm_low
            cur_file[torch.where(img >= (self.base_hu+hu_high))] = norm_high+self.base_norm
            # cur_file = (cur_file-torch.min(cur_file))/(torch.max(cur_file)-torch.min(cur_file))
            if j==0:
                new_file = cur_file
            else:
                new_file = torch.cat((new_file, cur_file), dim=1)

        return new_file

class Adapt_transform2(Module):
    """
    自适应Transform
    """

    def __init__(self, hu_lis=[0], norm_lis=[0], smooth_lis=[0], norm=False):
        super(Adapt_transform2, self).__init__()
        self.norm_lis = nn.Parameter(torch.FloatTensor(norm_lis))
        self.hu_lis = nn.Parameter(torch.FloatTensor(hu_lis))
        self.smooth_lis = nn.Parameter(torch.FloatTensor(smooth_lis))
        self.norm = norm

    def forward(self, img):
        """
        多个sigmoid转换后累加，注意此处hu值界限为叠加所得。
        :param img: tensor
        :param self.hu_lis:
        :param self.norm_lis:
        :param norm: norm or not
        :return:
        """
        for j in range(self.hu_lis.shape[0]):  # 对所有transform类遍历
            cur_file = torch.zeros_like(img)
            for i in range(self.hu_lis.shape[1]):  # 对一类中所有hu遍历
                cur_file += torch.abs(self.norm_lis[j,i])/(1+torch.exp((-img+100*torch.sum(self.hu_lis[j,0:i+1]))/(100*torch.abs(self.smooth_lis[j,i])+30)))
            if self.norm:
                cur_file = (cur_file-torch.min(cur_file))/(torch.max(cur_file)-torch.min(cur_file))
            if j == 0:
                new_file = cur_file
            else:
                new_file = torch.cat((new_file, cur_file), dim=1)
        return new_file

class Adapt_transform3(Module):
    """
    自适应Transform
    """

    def __init__(self, hu_lis=[0], norm_lis=[0], smooth_lis=[0], base_hu=-2000, base_norm=0, norm=False):
        super(Adapt_transform3, self).__init__()
        self.base_hu = base_hu
        self.base_norm = base_norm
        self.hu_lis = nn.Parameter(torch.FloatTensor(hu_lis))
        self.norm_lis = nn.Parameter(torch.FloatTensor(norm_lis))
        self.smooth_lis = nn.Parameter(torch.FloatTensor(smooth_lis))
        self.norm = norm

    def forward(self, img):
        """
        多个sigmoid转换后累加，注意此处hu值界限为直接赋值。
        :param img: tensor
        :param self.hu_lis:
        :param self.norm_lis:
        :param norm: norm or not
        :return:
        """
        for j in range(self.hu_lis.shape[0]):  # 对所有transform类遍历
            cur_file = torch.zeros_like(img)
            for i in range(self.hu_lis.shape[1]):  # 对一类中所有hu遍历
                cur_file += torch.abs(self.norm_lis[j,i])/(1+torch.exp((-img+100*self.hu_lis[j,i])/(100*torch.abs(self.smooth_lis[j,i])+30)))
            if self.norm:
                cur_file = (cur_file-torch.min(cur_file))/(torch.max(cur_file)-torch.min(cur_file))
            if j == 0:
                new_file = cur_file
            else:
                new_file = torch.cat((new_file, cur_file), dim=1)
        return new_file