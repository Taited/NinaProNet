from __future__ import print_function
from Training.models.module import Module
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from Training.models.layers import DoubleConv3D, SingleResSeparateConv3D, SingleResSeparateConv3D
import math

class RUnet_Separate_4(Module):
    def __init__(self,  inc=1, n_classes=5, base_chns=16, droprate=0, norm='in', depth=False, dilation=1):
        super(RUnet_Separate_4, self).__init__()
        self.model_name = "seg"
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')  # 1/4(h,h)
        self.downsample = nn.MaxPool3d(2, 2)  # 1/2(h,h)
        self.drop = nn.Dropout(droprate)

        self.conv1_1 = SingleResSeparateConv3D(2, base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv1_2 = SingleResSeparateConv3D(base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv2_1 = SingleResSeparateConv3D(2*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv2_2 = SingleResSeparateConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv3_1 = SingleResSeparateConv3D(4*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv3_2 = SingleResSeparateConv3D(4 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')


        self.conv5_1 = SingleResSeparateConv3D(8*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv5_2 = SingleResSeparateConv3D(8 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv6_1 = SingleResSeparateConv3D(12*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv6_2 = SingleResSeparateConv3D(4 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv7_1 = SingleResSeparateConv3D(7*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv7_2 = SingleResSeparateConv3D(2 * base_chns, 2 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.disconv_1 = SingleResSeparateConv3D(n_classes, base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')


        self.classification = nn.Sequential(
            nn.Dropout3d(p=0.1),
            nn.Conv3d(in_channels=2*base_chns, out_channels=n_classes, kernel_size=1),
        )


    def forward(self, x):
        out = self.conv1_1(x[:, 0:2])
        conv1 = self.conv1_2(out)
        out = self.downsample(conv1)  # 1/2
        out = self.conv2_1(out)
        conv2 = self.conv2_2(out)  #
        out = self.downsample(conv2)  # 1/4
        out = self.conv3_1(out)
        out = self.conv3_2(out)  #

        out = self.conv5_1(out)
        out = self.conv5_2(out)

        up6 = self.upsample(out)  # 1/2
        out = t.cat((up6, conv2), 1)
        out = self.conv6_1(out)
        out = self.conv6_2(out)

        dis_out = self.disconv_1(x[:, 2::])
        up7 = self.upsample(out)
        out = t.cat((up7, conv1, dis_out), 1)
        out = self.conv7_1(out)
        out = self.conv7_2(out)

        out = self.classification(out)
        predic = F.softmax(out, dim=1)
        return predic
