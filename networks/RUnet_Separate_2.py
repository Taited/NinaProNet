from __future__ import print_function
from .module import Module
import torch as t
import torch.nn as nn
from .layers import TriResSeparateConv3D
import torch.nn.functional as F
import math
class RUnet_Separate_2(Module):
    """
    3D_train Segmentation with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes
    中用的网络，相比普通U-net加入了res连接
    """
    def __init__(self, inc=1, n_classes=5, base_chns=48, droprate=0, norm='in', depth = False, dilation=1):
        super(RUnet_Separate_2, self).__init__()
        self.model_name = "seg"
        self.n_class = n_classes
        self.dropout = nn.Dropout(droprate)
        self.downsample = nn.MaxPool3d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')
        self.conv1 = TriResSeparateConv3D(inc, base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.conv2 = TriResSeparateConv3D(base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.conv3 = TriResSeparateConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.conv4 = TriResSeparateConv3D(4 * base_chns, 8 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.conv5 = TriResSeparateConv3D(8 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)

        self.conv6_1 = TriResSeparateConv3D(8 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.conv6_2 = TriResSeparateConv3D(4 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.conv7_1 = TriResSeparateConv3D(4 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.conv7_2 = TriResSeparateConv3D(2 * base_chns, 1 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.conv8_1 = TriResSeparateConv3D(2 * base_chns, base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.conv8_2 = TriResSeparateConv3D(base_chns, base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.dis_deep = nn.Conv3d(in_channels=n_classes, out_channels=n_classes, kernel_size=1)

        self.classification = nn.Sequential(
            nn.ReLU(),
            nn.Dropout3d(p=0.1),
            nn.Conv3d(in_channels=base_chns, out_channels=n_classes, kernel_size=1),
        )



    def forward(self, x):
        dis = x[:, -self.n_class::]
        conv1 = self.conv1(x)
        out = self.downsample(conv1)  # 1/2
        conv2 = self.conv2(out)  #
        out = self.downsample(conv2)  # 1/4
        conv3 = self.conv3(out)  #
        out = self.downsample(conv3)  # 1/8
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.dropout(out)

        out = self.upsample(out)  # 1/4
        out = t.cat((out, conv3), 1)
        out = self.conv6_1(out)
        out = self.conv6_2(out)

        out = self.upsample(out)    # 1/2
        out = t.cat((out, conv2), 1)
        out = self.conv7_1(out)
        out = self.conv7_2(out)

        out = self.upsample(out)  # 1/2
        out = t.cat((out, conv1), 1)
        out = self.conv8_1(out)
        out = self.conv8_2(out)

        out = self.classification(out)
        out = self.dis_deep(out * t.exp(dis)) + 2 * dis
        predic = F.softmax(out, dim=1)
        return predic
