from __future__ import print_function
from .module import Module
import torch as t
import torch.nn as nn
from .layers import TriResSeparateConv3D
import torch.nn.functional as F
import math
class UnetDense_Separate_5(Module):
    """
    相比普通U-net加入了res连接,只下采样1次，并分离了3D卷积为3*3*1+1*1*3,最终几层宽度增加
    """
    def __init__(self, inc=1, n_classes=5, base_chns=12, droprate=0, norm='in', depth = False, dilation=1):
        super(UnetDense_Separate_5, self).__init__()
        self.model_name = "seg"

        self.dropout = nn.Dropout(droprate)
        self.downsample = nn.MaxPool3d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

        self.conv1 = TriResSeparateConv3D(inc, 2*base_chns, norm=norm, depth=depth, pad='same', dilat=1)

        self.conv2 = TriResSeparateConv3D(2*base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)

        self.conv3 = TriResSeparateConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)

        self.DenseAspp = DenseASPP(4 * base_chns,  d_feature1=base_chns, norm=norm)

        self.conv6_1 = TriResSeparateConv3D(13 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.conv6_2 = TriResSeparateConv3D(4 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)

        self.conv7_1 = TriResSeparateConv3D(4 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)
        self.conv7_2 = TriResSeparateConv3D(2 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation)

        self.conv8_1 = TriResSeparateConv3D(4 * base_chns, 2*base_chns, norm=norm, depth=depth, pad='same', dilat=1)
        self.conv8_2 = TriResSeparateConv3D(2*base_chns, base_chns, norm=norm, depth=depth, pad='same', dilat=1)

        self.classification = nn.Sequential(
            nn.ReLU(),
            nn.Dropout3d(p=0.1),
            nn.Conv3d(in_channels=base_chns, out_channels=n_classes, kernel_size=1),
        )



    def forward(self, x):
        conv1 = self.conv1(x)
        out = self.downsample(conv1)  # 1/2
        conv2 = self.conv2(out)  #
        conv3 = self.conv3(out)  #

        out = self.DenseAspp(conv3)

        out = self.dropout(out)

        out = t.cat((out, conv3), 1)
        out = self.conv6_1(out)
        out = self.conv6_2(out)


        out = t.cat((out, conv2), 1)
        out = self.conv7_1(out)
        out = self.conv7_2(out)

        out = self.upsample(out)
        out = t.cat((out, conv1), 1)
        out = self.conv8_1(out)
        out = self.conv8_2(out)

        out = self.classification(out)
        predic = F.softmax(out, dim=1)
        return predic

class DenseASPP(nn.Module):
    def __init__(self, num_features, d_feature1, norm):
        super(DenseASPP, self).__init__()
        self.ASPP_3 = TriResSeparateConv3D(num_features, d_feature1 * 1, norm=norm, pad='same', dilat=3)

        self.ASPP_6 = TriResSeparateConv3D(num_features + d_feature1 * 1, d_feature1 , norm=norm, pad='same', dilat=3)

        self.ASPP_12 = TriResSeparateConv3D(num_features + d_feature1 * 2, d_feature1 , norm=norm, pad='same', dilat=3)

        self.ASPP_18 = TriResSeparateConv3D(num_features + d_feature1 * 3, d_feature1, norm=norm, pad='same', dilat=3)

        self.ASPP_24 = TriResSeparateConv3D(num_features + d_feature1 * 4, d_feature1, norm=norm, pad='same', dilat=3)



    def forward(self, input):
        aspp3 = self.ASPP_3(input)
        feature = t.cat((aspp3, input), dim=1)

        aspp6 = self.ASPP_6(feature)
        feature = t.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12(feature)
        feature = t.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18(feature)
        feature = t.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24(feature)
        feature = t.cat((aspp24, feature), dim=1)


        return feature