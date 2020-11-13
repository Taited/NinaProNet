from __future__ import print_function
from networks.module import Module
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from networks.layers import SingleConv3D, DenseSingleSeparateConv3D, Transition
import math

class DenseSepUnet(Module):
    def __init__(self,  inc=1, n_classes=5, base_chns=16, droprate=0, norm='in', depth=False, dilation=1):
        super(DenseSepUnet, self).__init__()
        self.model_name = "seg"
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')  # 1/4(h,h)
        self.downsample = nn.MaxPool3d(2, 2)  # 1/2(h,h)
        self.drop = nn.Dropout(droprate)

        self.conv1_1 = SingleConv3D(inc, base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv1_2 = DenseSingleSeparateConv3D(base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv1_3 = Transition(3*base_chns, 2*base_chns, norm=norm, depth=depth)

        self.conv2_1 = DenseSingleSeparateConv3D(2*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv2_2 = DenseSingleSeparateConv3D(4 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv2_3 = Transition(8 * base_chns, 4 * base_chns, norm=norm, depth=depth)

        self.conv3_1 = DenseSingleSeparateConv3D(4*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv3_2 = DenseSingleSeparateConv3D(8 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv3_3 = Transition(16 * base_chns, 8 * base_chns, norm=norm, depth=depth)

        self.conv4_1 = DenseSingleSeparateConv3D(8*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=math.ceil(dilation/2), pad='same')
        self.conv4_2 = DenseSingleSeparateConv3D(16 * base_chns, 16 * base_chns, norm=norm, depth=depth, dilat=math.ceil(dilation/2), pad='same')
        self.conv4_3 = Transition(32 * base_chns, 8 * base_chns, norm=norm, depth=depth)

        self.conv5_1 = DenseSingleSeparateConv3D(16*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv5_2 = DenseSingleSeparateConv3D(24 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv5_3 = Transition(32 * base_chns, 8 * base_chns, norm=norm, depth=depth)

        self.conv6_1 = DenseSingleSeparateConv3D(12*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv6_2 = DenseSingleSeparateConv3D(16 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv6_3 = Transition(20 * base_chns, 4 * base_chns, norm=norm, depth=depth)

        self.conv7_1 = DenseSingleSeparateConv3D(6*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv7_2 = DenseSingleSeparateConv3D(8 * base_chns, 2 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv7_3 = Transition(10 * base_chns, 2 * base_chns, norm=norm, depth=depth)

        self.classification = nn.Sequential(
            nn.Dropout3d(p=0.1),
            nn.Conv3d(in_channels=2*base_chns, out_channels=n_classes, kernel_size=1),
        )


    def forward(self, x):
        out = self.conv1_1(x)
        out = self.conv1_2(out)
        conv1 = self.conv1_3(out)
        out = self.downsample(conv1)  # 1/2

        out = self.conv2_1(out)
        out = self.conv2_2(out)
        conv2 = self.conv2_3(out)  #
        out = self.downsample(conv2)  # 1/4

        out = self.conv3_1(out)
        out = self.conv3_2(out)  #
        conv3 = self.conv3_3(out)  #
        out = self.downsample(conv3)  # 1/8

        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.conv4_3(out)
        out = self.drop(out)

        up5 = self.upsample(out)  # 1/4
        out = t.cat((up5, conv3), 1)
        out = self.conv5_1(out)
        out = self.conv5_2(out)
        out = self.conv5_3(out)

        up6 = self.upsample(out)  # 1/2
        out = t.cat((up6, conv2), 1)
        out = self.conv6_1(out)
        out = self.conv6_2(out)
        out = self.conv6_3(out)

        up7 = self.upsample(out)
        out = t.cat((up7, conv1), 1)
        out = self.conv7_1(out)
        out = self.conv7_2(out)
        out = self.conv7_3(out)

        out = self.classification(out)
        predic = F.softmax(out, dim=1)
        return predic
