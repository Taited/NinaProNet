from __future__ import print_function
from networks.module import Module
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from networks.layers import *
import math

class MSnet(Module):
    '''
    初始版本的MultiSpacingNetwork，low pixel feature map经上采样后与middle pixel feature map concat，
    concat后feature map经1*1*1卷积后上采样再与high pixel feature map concat。
    '''
    def __init__(self,  inc=1, n_classes=5, base_chns=16, droprate=0, norm='in', depth=False, dilation=1):
        super(MSnet, self).__init__()
        self.model_name = "seg"
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')  # 1/4(h,h)
        self.downsample = nn.MaxPool3d(2, 2)  # 1/2(h,h)
        self.interpolation = nn.Upsample(scale_factor=(2,1,1), mode='trilinear')
        self.drop = nn.Dropout(droprate)

        self.conv1_1_lr = SingleConv3D(inc, base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv1_2_lr = SingleConv3D(base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv1_1_mr = SingleConv3D(inc, base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv1_2_mr = SingleConv3D(base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv1_3_mr = depthwise(4*base_chns, 2 * base_chns, kernel_size=1, depth=depth, padding=0)
        self.conv1_1_hr = SingleConv3D(inc, base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv1_2_hr = SingleConv3D(base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv1_3_hr = depthwise(4 * base_chns, 2 * base_chns, kernel_size=1, depth=depth, padding=0)


        self.conv2_1_lr = SingleConv3D(2*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv2_2_lr = SingleConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv2_1_mr = SingleConv3D(2*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv2_2_mr = SingleConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv2_3_mr = depthwise(8*base_chns, 4 * base_chns, kernel_size=1, depth=depth, padding=0)
        self.conv2_1_hr = SingleConv3D(2*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv2_2_hr = SingleConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv2_3_hr = depthwise(8*base_chns, 4 * base_chns, kernel_size=1, depth=depth, padding=0)


        self.conv3_1_lr = SingleConv3D(4*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv3_2_lr = SingleConv3D(4 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv3_1_mr = SingleConv3D(4*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv3_2_mr = SingleConv3D(4 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv3_3_mr = depthwise(8*base_chns, 4 * base_chns, kernel_size=1, depth=depth, padding=0)
        self.conv3_1_hr = SingleConv3D(4*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv3_2_hr = SingleConv3D(4 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv3_3_hr = depthwise(8*base_chns, 4 * base_chns, kernel_size=1, depth=depth, padding=0)

        self.conv4_1_lr = SingleConv3D(4*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=math.ceil(dilation/2), pad='same')
        self.conv4_2_lr = SingleConv3D(8 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=math.ceil(dilation/2), pad='same')
        self.conv4_1_mr = SingleConv3D(4*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=math.ceil(dilation/2), pad='same')
        self.conv4_2_mr = SingleConv3D(8 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=math.ceil(dilation/2), pad='same')
        self.conv4_3_mr = depthwise(16 * base_chns, 8 * base_chns, kernel_size=1, depth=depth, padding=0)
        self.conv4_1_hr = SingleConv3D(4*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=math.ceil(dilation/2), pad='same')
        self.conv4_2_hr = SingleConv3D(8 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=math.ceil(dilation/2), pad='same')
        self.conv4_3_hr = depthwise(16 * base_chns, 8 * base_chns, kernel_size=1, depth=depth, padding=0)


        self.conv5_1_lr = SingleConv3D(12*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv5_2_lr = SingleConv3D(8 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv5_1_mr = SingleConv3D(12*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv5_2_mr = SingleConv3D(8 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv5_3_mr = depthwise(16 * base_chns, 8 * base_chns, kernel_size=1, depth=depth, padding=0)
        self.conv5_1_hr = SingleConv3D(12*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv5_2_hr = SingleConv3D(8 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv5_3_hr = depthwise(16 * base_chns, 8 * base_chns, kernel_size=1, depth=depth, padding=0)

        self.conv6_1_lr = SingleConv3D(12*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv6_2_lr = SingleConv3D(4 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv6_1_mr = SingleConv3D(12*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv6_2_mr = SingleConv3D(4 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv6_3_mr = depthwise(8 * base_chns, 4 * base_chns, kernel_size=1, depth=depth, padding=0)
        self.conv6_1_hr = SingleConv3D(12*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv6_2_hr = SingleConv3D(4 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv6_3_hr = depthwise(8 * base_chns, 4 * base_chns, kernel_size=1, depth=depth, padding=0)

        self.conv7_1_lr = SingleConv3D(6*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv7_2_lr = SingleConv3D(2 * base_chns, 2 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv7_1_mr = SingleConv3D(6*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv7_2_mr = SingleConv3D(2 * base_chns, 2 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv7_3_mr = depthwise(4 * base_chns, 2 * base_chns, kernel_size=1, depth=depth, padding=0)
        self.conv7_1_hr = SingleConv3D(6*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv7_2_hr = SingleConv3D(2 * base_chns, 2 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv7_3_hr = depthwise(4 * base_chns, 2 * base_chns, kernel_size=1, depth=depth, padding=0)

        self.classification_lr = nn.Sequential(
            nn.Conv3d(in_channels=2*base_chns, out_channels=n_classes, kernel_size=1),
        )
        self.classification_mr = nn.Sequential(
            nn.Conv3d(in_channels=2*base_chns, out_channels=n_classes, kernel_size=1),
        )
        self.classification_hr = nn.Sequential(
            nn.Conv3d(in_channels=2*base_chns, out_channels=n_classes, kernel_size=1),
        )


    def forward(self, x_hr, x_mr, x_lr):
        out_lr = self.conv1_1_lr(x_lr)
        conv1_lr = self.conv1_2_lr(out_lr)
        out_lr = self.downsample(conv1_lr)  # 1/2
        out_mr = self.conv1_1_mr(x_mr)
        out_mr = self.conv1_2_mr(out_mr)
        conv1_mr = self.conv1_3_mr(t.cat((self.interpolation(conv1_lr), out_mr),1))
        out_mr = self.downsample(conv1_mr)  # 1/2
        out_hr = self.conv1_1_hr(x_hr)
        out_hr = self.conv1_2_hr(out_hr)
        conv1_hr = self.conv1_3_hr(t.cat((self.interpolation(conv1_mr), out_hr),1))
        out_hr = self.downsample(conv1_hr)  # 1/2

        out_lr = self.conv2_1_lr(out_lr)
        conv2_lr = self.conv2_2_lr(out_lr)
        out_lr = self.downsample(conv2_lr)  # 1/4
        out_mr = self.conv2_1_mr(out_mr)
        out_mr = self.conv2_2_mr(out_mr)
        conv2_mr = self.conv2_3_mr(t.cat((self.interpolation(conv2_lr), out_mr),1))
        out_mr = self.downsample(conv2_mr)  # 1/4
        out_hr = self.conv2_1_hr(out_hr)
        out_hr = self.conv2_2_hr(out_hr)
        conv2_hr = self.conv2_3_hr(t.cat((self.interpolation(conv2_mr), out_hr),1))
        out_hr = self.downsample(conv2_hr)  # 1/4

        out_lr = self.conv3_1_lr(out_lr)
        conv3_lr = self.conv3_2_lr(out_lr)
        out_lr = self.downsample(conv3_lr)  # 1/8
        out_mr = self.conv3_1_mr(out_mr)
        out_mr = self.conv3_2_mr(out_mr)
        conv3_mr = self.conv3_3_mr(t.cat((self.interpolation(conv3_lr), out_mr),1))
        out_mr = self.downsample(conv3_mr)  # 1/8
        out_hr = self.conv3_1_hr(out_hr)
        out_hr = self.conv3_2_hr(out_hr)
        conv3_hr = self.conv3_3_hr(t.cat((self.interpolation(conv3_mr), out_hr),1))
        out_hr = self.downsample(conv3_hr)  # 1/8

        out_lr = self.conv4_1_lr(out_lr)
        out_lr = self.conv4_2_lr(out_lr)
        out_mr = self.conv4_1_mr(out_mr)
        out_mr = self.conv4_2_mr(out_mr)
        out_mr = self.conv4_3_mr(t.cat((self.interpolation(out_lr), out_mr),1))
        out_hr = self.conv4_1_hr(out_hr)
        out_hr = self.conv4_2_hr(out_hr)
        out_hr = self.conv4_3_hr(t.cat((self.interpolation(out_mr), out_hr),1))

        out_lr = self.drop(out_lr)  # 1/2
        out_mr = self.drop(out_mr)  # 1/2
        out_hr = self.drop(out_hr)  # 1/2

        out_lr = self.upsample(out_lr)
        out_lr = t.cat((out_lr, conv3_lr), 1)
        out_lr = self.conv5_1_lr(out_lr)
        out_lr = self.conv5_2_lr(out_lr)
        out_mr = self.upsample(out_mr)
        out_mr = t.cat((out_mr, conv3_mr), 1)
        out_mr = self.conv5_1_mr(out_mr)
        out_mr = self.conv5_2_mr(out_mr)
        out_mr = self.conv5_3_mr(t.cat((self.interpolation(out_lr), out_mr),1))
        out_hr = self.upsample(out_hr)
        out_hr = t.cat((out_hr, conv3_hr), 1)
        out_hr = self.conv5_1_hr(out_hr)
        out_hr = self.conv5_2_hr(out_hr)
        out_hr = self.conv5_3_hr(t.cat((self.interpolation(out_mr), out_hr),1))

        out_lr = self.upsample(out_lr)
        out_lr = t.cat((out_lr, conv2_lr), 1)
        out_lr = self.conv6_1_lr(out_lr)
        out_lr = self.conv6_2_lr(out_lr)
        out_mr = self.upsample(out_mr)
        out_mr = t.cat((out_mr, conv2_mr), 1)
        out_mr = self.conv6_1_mr(out_mr)
        out_mr = self.conv6_2_mr(out_mr)
        out_mr = self.conv6_3_mr(t.cat((self.interpolation(out_lr), out_mr),1))
        out_hr = self.upsample(out_hr)
        out_hr = t.cat((out_hr, conv2_hr), 1)
        out_hr = self.conv6_1_hr(out_hr)
        out_hr = self.conv6_2_hr(out_hr)
        out_hr = self.conv6_3_hr(t.cat((self.interpolation(out_mr), out_hr),1))

        out_lr = self.upsample(out_lr)
        out_lr = t.cat((out_lr, conv1_lr), 1)
        out_lr = self.conv7_1_lr(out_lr)
        out_lr = self.conv7_2_lr(out_lr)
        out_mr = self.upsample(out_mr)
        out_mr = t.cat((out_mr, conv1_mr), 1)
        out_mr = self.conv7_1_mr(out_mr)
        out_mr = self.conv7_2_mr(out_mr)
        out_mr = self.conv7_3_mr(t.cat((self.interpolation(out_lr), out_mr),1))
        out_hr = self.upsample(out_hr)
        out_hr = t.cat((out_hr, conv1_hr), 1)
        out_hr = self.conv7_1_hr(out_hr)
        out_hr = self.conv7_2_hr(out_hr)
        out_hr = self.conv7_3_hr(t.cat((self.interpolation(out_mr), out_hr),1))

        out_lr = self.classification_lr(out_lr)
        out_mr = self.classification_mr(out_mr)
        out_hr = self.classification_hr(out_hr)
        predic = {'predic_lr':F.softmax(out_lr, dim=1),'predic_mr':F.softmax(out_mr, dim=1),'predic_hr':F.softmax(out_hr, dim=1)}
        return predic
