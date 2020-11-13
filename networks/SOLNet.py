from __future__ import print_function
from networks.module import Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers import SEResBlock3D, DenseAsppBlock3D

class SOLNet(Module):
    def __init__(self, inc=1, n_classes=5, base_chns=16, droprate=0, norm='in', depth=False, dilation=1):
        super(SOLNet, self).__init__()
        self.SEResBlock_Bridge_1 = SEResBlock3D(cin=inc,cout=base_chns, norm=norm)
        self.SEResBlock_1_1 = SEResBlock3D(cin=inc, cout=base_chns, norm=norm)
        self.SEResBlock_1_2 = SEResBlock3D(cin=base_chns, cout=base_chns, norm=norm)
        self.DownPool = nn.MaxPool3d(2)

        self.SEResBlock_Bridge_2 = SEResBlock3D(cin=base_chns, cout=2*base_chns, norm=norm)
        self.SEResBlock_2_1 = SEResBlock3D(cin=base_chns, cout=2*base_chns, norm=norm, dilation=dilation)
        self.SEResBlock_2_2 = SEResBlock3D(cin=2*base_chns, cout=2*base_chns, norm=norm, dilation=dilation)

        self.DenseAspp = DenseASPP(num_features=2*base_chns, d_feature0=base_chns, d_feature1=base_chns, dropout=droprate, norm=norm)

        self.SEResBlock_3 = SEResBlock3D(cin=7*base_chns, cout=4*base_chns, norm=norm)
        self.SEResBlock_4 = SEResBlock3D(cin=6*base_chns, cout=4*base_chns, norm=norm)

        self.UpPool = nn.Upsample(scale_factor=2, mode='trilinear')

        self.SEResBlock_5 = SEResBlock3D(cin=5*base_chns, cout=2*base_chns, norm=norm)

        self.classfication = nn.Sequential(
            nn.ReLU(),
            nn.Dropout3d(p=0.1),
            nn.Conv3d(in_channels=2*base_chns, out_channels=n_classes, kernel_size=1))

    def forward(self, input):
        Feature1 = self.SEResBlock_Bridge_1(input)
        out = self.SEResBlock_1_1(input)
        out = self.SEResBlock_1_2(out)
        out = self.DownPool(out)
        Feature2 = self.SEResBlock_Bridge_2(out)
        out = self.SEResBlock_2_1(out)
        out = self.SEResBlock_2_2(out)
        out = self.DenseAspp(out)
        out = self.SEResBlock_3(out)
        out = torch.cat((out, Feature2), dim=1)
        out = self.SEResBlock_4(out)
        out = self.UpPool(out)
        out = torch.cat((out, Feature1), dim=1)
        out = self.SEResBlock_5(out)
        out = self.classfication(out)

        prediction = F.softmax(out, dim=1)

        return prediction






class DenseASPP(nn.Module):
    def __init__(self, num_features, d_feature0, d_feature1, dropout, norm):
        super(DenseASPP, self).__init__()
        self.ASPP_3 = DenseAsppBlock3D(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=3, drop_out=dropout, norm=norm, norm_start=True)

        self.ASPP_6 = DenseAsppBlock3D(input_num=num_features + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=6, drop_out=dropout, norm=norm, norm_start=True)

        self.ASPP_12 = DenseAsppBlock3D(input_num=num_features + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=12, drop_out=dropout, norm=norm, norm_start=True)

        self.ASPP_18 = DenseAsppBlock3D(input_num=num_features + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=18, drop_out=dropout, norm=norm, norm_start=True)

        self.ASPP_24 = DenseAsppBlock3D(input_num=num_features + d_feature1 * 4, num1=d_feature0, num2=d_feature1,
                                    dilation_rate=24, drop_out=dropout, norm=norm, norm_start=True)



    def forward(self, input):
        aspp3 = self.ASPP_3(input)
        feature = torch.cat((aspp3, input), dim=1)

        aspp6 = self.ASPP_6(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24(feature)
        feature = torch.cat((aspp24, feature), dim=1)


        return feature