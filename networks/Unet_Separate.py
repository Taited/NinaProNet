from __future__ import print_function
from networks.module import Module
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from networks.layers import QuadResSeparateConv3D, TriSeparateConv3D


class Unet_Separate(Module):
    def __init__(self,  inc=1, n_classes=5, base_chns=16, droprate=0, norm='in', depth=False, dilation=1):
        super(Unet_Separate, self).__init__()
        self.model_name = "seg"
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')
        self.downsample = nn.MaxPool3d(2, 2)
        self.dropout = nn.Dropout(droprate)


        self.conv1 = QuadResSeparateConv3D(inc, base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.conv2 = QuadResSeparateConv3D(base_chns, 2*base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.conv3 = QuadResSeparateConv3D(2*base_chns, 4*base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.conv4_1 = TriSeparateConv3D(4*base_chns, 8*base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.conv4_2 = TriSeparateConv3D(8*base_chns, 4*base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)


        self.conv5_1 = TriSeparateConv3D(8*base_chns, 4*base_chns, norm=norm, depth=depth)
        self.conv5_2 = TriSeparateConv3D(4*base_chns, 2*base_chns, norm=norm, depth=depth)
        self.conv6_1 = TriSeparateConv3D(4*base_chns, 2*base_chns, norm=norm, depth=depth)
        self.conv6_2 = TriSeparateConv3D(2*base_chns, base_chns, norm=norm, depth=depth)
        self.conv7_1 = QuadResSeparateConv3D(2*base_chns, base_chns, norm=norm, depth=depth)
        self.conv7_2 = TriSeparateConv3D(base_chns, base_chns, norm=norm, depth=depth)

        self.classification = nn.Sequential(
            nn.ReLU(),
            nn.Dropout3d(p=0.1),
            nn.Conv3d(in_channels=base_chns, out_channels=n_classes, kernel_size=1),
        )


    def forward(self, x):
        conv1 = self.conv1(x)
        out = self.downsample(conv1)  # 1/2
        conv2 = self.conv2(out)  #
        out = self.downsample(conv2)  # 1/4
        conv3 = self.conv3(out)  #
        out = self.downsample(conv3)  # 1/8
        out = self.conv4_1(out)
        out = self.conv4_2(out)

        out = self.dropout(out)

        up5 = self.upsample(out)  # 1/4
        out = t.cat((up5, conv3), 1)
        out = self.conv5_1(out)
        out = self.conv5_2(out)

        up6 = self.upsample(out)  # 1/2
        out = t.cat((up6, conv2), 1)
        out = self.conv6_1(out)
        out = self.conv6_2(out)

        up7 = self.upsample(out)
        out = t.cat((up7, conv1), 1)
        out = self.conv7_1(out)
        out = self.conv7_2(out)

        out = self.classification(out)
        predic = F.softmax(out, dim=1)
        return predic
