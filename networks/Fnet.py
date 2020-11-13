from __future__ import print_function
from networks.module import Module
import torch as t
import torch.nn as nn
from networks.layers import TriResSeparateConv3D
import torch.nn.functional as F


class Fnet(Module):
    """
    提取特征
    """
    def __init__(self, inc=1, base_chns=12, droprate=0, norm='in', depth = False, dilation=1, separate_direction='axial'):
        super(Fnet, self).__init__()
        self.model_name = "seg"

        self.dropout = nn.Dropout(droprate)
        self.downsample = nn.MaxPool3d(2, 2)

        self.conv1 = TriResSeparateConv3D(inc, 2*base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)

        self.conv2 = TriResSeparateConv3D(2*base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)

        self.conv3 = TriResSeparateConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)


        self.conv4 = TriResSeparateConv3D(4 * base_chns, 8 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv5 = TriResSeparateConv3D(8 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)




    def forward(self, x):
        conv1 = self.conv1(x)
        out = self.downsample(conv1)  # 1/2
        conv2 = self.conv2(out)  #
        out = self.downsample(conv2)  # 1/4
        conv3 = self.conv3(out)  #
        out = self.downsample(conv3)  # 1/8
        out = self.dropout(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out
