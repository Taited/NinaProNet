from __future__ import print_function
from networks.module import Module
import torch as t
import torch.nn as nn
from networks.layers import QuadResSeparateConv3D
import torch.nn.functional as F


class Unet_Separate_2(Module):
    """
    相比普通U-net加入了res连接,并分离了3D卷积为3*3*1+1*1*3,最终几层宽度增加
    """
    def __init__(self, inc=1, n_classes=5, base_chns=12, droprate=0, norm='in', depth = False, dilation=1, separate_direction='axial',output_feature=False):
        super(Unet_Separate_2, self).__init__()
        self.model_name = "seg"
        self.output_feature=output_feature
        self.dropout = nn.Dropout(droprate)
        self.downsample = nn.MaxPool3d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

        self.conv1 = QuadResSeparateConv3D(inc, 2*base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)

        self.conv2 = QuadResSeparateConv3D(2*base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)

        self.conv3 = QuadResSeparateConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)


        self.conv4 = QuadResSeparateConv3D(4 * base_chns, 8 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv5 = QuadResSeparateConv3D(8 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)


        self.conv6_1 = QuadResSeparateConv3D(8 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv6_2 = QuadResSeparateConv3D(4 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)

        self.conv7_1 = QuadResSeparateConv3D(4 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv7_2 = QuadResSeparateConv3D(2 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)

        self.conv8_1 = QuadResSeparateConv3D(4 * base_chns, 2*base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv8_2 = QuadResSeparateConv3D(2*base_chns, 2*base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)

        self.classification = nn.Sequential(
            nn.ReLU(),
            nn.Dropout3d(p=0.1),
            nn.Conv3d(in_channels=2*base_chns, out_channels=n_classes, kernel_size=1),
        )



    def forward(self, x):
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
        conv8 = self.conv8_2(out)

        out = self.classification(conv8)
        predic = F.softmax(out, dim=1)
        if  self.output_feature:
            return predic,t.cat((out, conv8,conv1),1)
        else:
            return predic
