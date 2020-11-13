from __future__ import print_function
from networks.module import Module
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from networks.layers import BasicResConv3D, SingleConv3D
from networks.cbam import CBAM

class Unet_CBAM(Module):
    def __init__(self,  inc=1, n_classes=5, base_chns=16, droprate=0, norm='in', depth=False):
        super(Unet_CBAM, self).__init__()
        self.model_name = "seg"

        self.conv1 = BasicResConv3D(inc, base_chns, droprate=droprate, norm=norm, depth=depth)
        self.cbam1 = CBAM(base_chns, 8)
        self.downsample1 = nn.MaxPool3d(2, 2)  # 1/2(h,h)

        self.conv2 = BasicResConv3D(base_chns, 2*base_chns, droprate=droprate, norm=norm, depth=depth)
        self.cbam2 = CBAM(2*base_chns, 8)
        self.downsample2 = nn.MaxPool3d(2, 2)  # 1/4(h,h)

        self.conv3 = BasicResConv3D(2*base_chns, 4*base_chns, droprate=droprate, norm=norm, depth=depth)
        self.cbam3 = CBAM(4 * base_chns, 8)
        self.downsample3 = nn.MaxPool3d(2, 2)  # 1/8(h,h)

        self.conv4_1 = SingleConv3D(4*base_chns, 8*base_chns, norm=norm, depth=depth)
        self.cbam4_1 = CBAM(8 * base_chns, 8)
        self.conv4_2 = SingleConv3D(8*base_chns, 4*base_chns, norm=norm, depth=depth)
        self.cbam4_2 = CBAM(4 * base_chns, 8)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear')  # 1/4(h,h)
        self.conv5_1 = SingleConv3D(8*base_chns, 4*base_chns, norm=norm, depth=depth)
        self.conv5_2 = SingleConv3D(4*base_chns, 2*base_chns, norm=norm, depth=depth)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear')  # 1/2(h,h)
        self.conv6_1 = SingleConv3D(4*base_chns, 2*base_chns, norm=norm, depth=depth)
        self.conv6_2 = SingleConv3D(2*base_chns, base_chns, norm=norm, depth=depth)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear')  # (h,h)
        self.conv7_1 = BasicResConv3D(2*base_chns, base_chns, norm=norm, depth=depth)
        self.conv7_2 = SingleConv3D(base_chns, base_chns, norm=norm, depth=depth)

        self.classification = nn.Sequential(
            nn.ReLU(),
            nn.Dropout3d(p=0.1),
            nn.Conv3d(in_channels=base_chns, out_channels=n_classes, kernel_size=1),
        )


    def forward(self, x):
        conv1 = self.conv1(x)
        cbam1 = self.cbam1(conv1)
        down1 = self.downsample1(cbam1)  # 1/2
        conv2 = self.conv2(down1)  #
        cbam2 = self.cbam2(conv2)
        down2 = self.downsample2(cbam2)  # 1/4
        conv3 = self.conv3(down2)  #
        cbam3 = self.cbam3(conv3)
        down3 = self.downsample3(cbam3)  # 1/8
        conv4_1 = self.conv4_1(down3)
        cbam4_1 = self.cbam4_1(conv4_1)
        conv4_2 = self.conv4_2(cbam4_1)
        cbam4_2 = self.cbam4_2(conv4_2)


        up5 = self.upsample1(cbam4_2)  # 1/4
        cat5 = t.cat((up5, conv3), 1)
        conv5_1 = self.conv5_1(cat5)
        conv5_2 = self.conv5_2(conv5_1)

        up6 = self.upsample2(conv5_2)  # 1/2
        cat6 = t.cat((up6, conv2), 1)
        conv6_1 = self.conv6_1(cat6)
        conv6_2 = self.conv6_2(conv6_1)

        up7 = self.upsample3(conv6_2)
        cat7 = t.cat((up7, conv1), 1)
        conv7_1 = self.conv7_1(cat7)
        conv7_2 = self.conv7_2(conv7_1)

        cls = self.classification(conv7_2)
        predic = F.softmax(cls, dim=1)
        return predic
