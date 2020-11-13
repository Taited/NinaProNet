from __future__ import print_function
from networks.module import Module
import torch as t
import torch.nn as nn
from networks.layers import TriResSeparateConv3D
import torch.nn.functional as F


class DS_Unet_Separate_4(Module):
    """
    相比普通U-net加入了res连接,并分离了3D卷积为3*3*1+1*1*3,最终几层宽度增加
    """
    def __init__(self, inc=1, n_classes=5, base_chns=12, droprate=0, norm='in', depth = False, dilation=1):
        super(DS_Unet_Separate_4, self).__init__()
        self.model_name = "seg"

        self.dropout = nn.Dropout(droprate)
        self.downsample = nn.MaxPool3d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

        self.conv1 = TriResSeparateConv3D(inc, 2*base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)

        self.conv2 = TriResSeparateConv3D(2*base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)

        self.conv3 = TriResSeparateConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)


        self.conv4 = TriResSeparateConv3D(4 * base_chns, 8 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.conv5 = TriResSeparateConv3D(8 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)


        self.conv6_1 = TriResSeparateConv3D(8 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.conv6_2 = TriResSeparateConv3D(4 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)

        self.conv7_1 = TriResSeparateConv3D(4 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, active=False)
        self.conv7_2 = TriResSeparateConv3D(2 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, active=False)

        self.conv8_1 = TriResSeparateConv3D(4 * base_chns, 2*base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, active=False)
        self.conv8_2 = TriResSeparateConv3D(2*base_chns, 2*base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, active=False)


        self.classification1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.ReLU(),
            nn.Dropout3d(p=0.1),
            TriResSeparateConv3D(4 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation),
            nn.Conv3d(in_channels=2 * base_chns, out_channels=n_classes, kernel_size=1)
        )

        self.classification2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout3d(p=0.1),
            TriResSeparateConv3D(4 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation),
            nn.Conv3d(in_channels=2 * base_chns, out_channels=n_classes, kernel_size=1)
        )

        self.classification3 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout3d(p=0.1),
            nn.Conv3d(in_channels=2*base_chns, out_channels=n_classes, kernel_size=1)
        )





    def forward(self, x, mode):
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
        predic1 = F.softmax(self.classification1(out), dim=1)
        out = self.conv7_1(out)
        out = self.conv7_2(out)

        out = self.upsample(out)  # 1/2
        out = t.cat((out, conv1), 1)
        predic2 = F.softmax(self.classification2(out), dim=1)
        out = self.conv8_1(out)
        out = self.conv8_2(out)

        out = self.classification3(out)
        predic3 = F.softmax(out, dim=1)

        if mode =='train':
            return [predic1, predic2, predic3]
        else:
            return predic3
