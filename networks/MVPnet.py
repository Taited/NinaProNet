from __future__ import print_function
from networks.module import Module
import torch as t
import torch.nn as nn
from networks.layers import TriResSeparateConv3D
import torch.nn.functional as F


class MVPnet(Module):
    """
    相比普通U-net加入了res连接,并分离了3D卷积为3*3*1+1*1*3,最终几层宽度增加
    """
    def __init__(self, inc=1, n_classes=5, base_chns=12, droprate=0, norm='in', depth = False, dilation=1, separate_direction='axial'):
        super(MVPnet, self).__init__()
        self.model_name = "seg"
        self.dropout = nn.Dropout(droprate)
        self.downsample = nn.MaxPool3d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

        self.conv1_v1 = TriResSeparateConv3D(1, 2*base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv2_v1 = TriResSeparateConv3D(2*base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv3_v1 = TriResSeparateConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv1_v2 = TriResSeparateConv3D(1, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation,separate_direction=separate_direction)
        self.conv2_v2 = TriResSeparateConv3D(2 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same',dilat=dilation, separate_direction=separate_direction)
        self.conv3_v2 = TriResSeparateConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same',dilat=dilation, separate_direction=separate_direction)
        self.conv1_v3= TriResSeparateConv3D(1, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation,separate_direction=separate_direction)
        self.conv2_v3 = TriResSeparateConv3D(2 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same',dilat=dilation, separate_direction=separate_direction)
        self.conv3_v3 = TriResSeparateConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same',dilat=dilation, separate_direction=separate_direction)
        self.conv4 = TriResSeparateConv3D(12 * base_chns, 8 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv5 = TriResSeparateConv3D(8 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv6_1 = TriResSeparateConv3D(16 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv6_2 = TriResSeparateConv3D(4 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv7_1 = TriResSeparateConv3D( 8* base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv7_2 = TriResSeparateConv3D(2 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv8_1 = TriResSeparateConv3D(8 * base_chns, 2*base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv8_2 = TriResSeparateConv3D(2*base_chns, 2*base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.classification = nn.Sequential(
            nn.ReLU(),
            nn.Dropout3d(p=0.1),
            nn.Conv3d(in_channels=2*base_chns, out_channels=n_classes, kernel_size=1),
        )


    def forward(self, x):
        conv1_v1 = self.conv1_v1(x[:, 0:1])
        out = self.downsample(conv1_v1)  # 1/2
        conv2_v1 = self.conv2_v1(out)  #
        out = self.downsample(conv2_v1)  # 1/4
        conv3_v1 = self.conv3_v1(out)  #
        out_v1 = self.downsample(conv3_v1)  # 1/8
        conv1_v2 = self.conv1_v2(x[:, 1:2])
        out = self.downsample(conv1_v2)  # 1/2
        conv2_v2 = self.conv2_v2(out)  #
        out = self.downsample(conv2_v2)  # 1/4
        conv3_v2 = self.conv3_v2(out)  #
        out_v2 = self.downsample(conv3_v2)  # 1/8

        conv1_v3 = self.conv1_v3(x[:, -1::])
        out = self.downsample(conv1_v3)  # 1/2
        conv2_v3 = self.conv2_v3(out)  #
        out = self.downsample(conv2_v3)  # 1/4
        conv3_v3 = self.conv3_v3(out)  #
        out_v3 = self.downsample(conv3_v3)  # 1/8
        out = t.cat((out_v1, out_v2, out_v3), 1)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.dropout(out)
        out = self.upsample(out)  # 1/4
        out = t.cat((out, conv3_v1, conv3_v2, conv3_v3), 1)
        out = self.conv6_1(out)
        out = self.conv6_2(out)
        out = self.upsample(out)    # 1/2
        out = t.cat((out, conv2_v1, conv2_v2, conv2_v3), 1)
        out = self.conv7_1(out)
        out = self.conv7_2(out)
        out = self.upsample(out)  # 1/2
        conv8 = t.cat((out, conv1_v1, conv1_v2, conv1_v3), 1)
        out = self.conv8_1(conv8)
        out = self.conv8_2(out)
        out = self.classification(out)
        predic = F.softmax(out, dim=1)
        return predic
