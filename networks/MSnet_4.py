from __future__ import print_function
from networks.module import Module
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from networks.layers import *
import math

class MSnet(Module):
    '''
    MultiSpacingNetwork，low/high pixel feature map 均经上下采样后concat，
    '''
    def __init__(self,  inc=1, n_classes=5, base_chns=16, droprate=0, norm='in', depth=False, dilation=1):
        super(MSnet, self).__init__()
        self.model_name = "seg"
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')  # 1/4(h,h)
        self.downsample = nn.MaxPool3d(2, 2)  # 1/2(h,h)
        self.upinter_2 = nn.Upsample(scale_factor=(2,1,1), mode='trilinear')
        self.downinter_2 = nn.MaxPool3d(kernel_size=(2,1,1),stride=(2,1,1))
        self.drop = nn.Dropout(droprate)

        self.conv1_1_lr = SingleConv3D(inc, base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv1_2_lr = SingleConv3D(base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv1_3_lr = depthwise(4* base_chns, 2 * base_chns, kernel_size=1, depth=depth, padding=0)
        self.conv1_1_mr = SingleResConv3D(inc, base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv1_2_mr = SingleResConv3D(base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv1_3_mr = depthwise(4*base_chns, 2 * base_chns, kernel_size=1, depth=depth, padding=0)


        self.conv2_1_lr = SingleConv3D(2*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv2_2_lr = SingleConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv2_3_lr = depthwise(8 * base_chns, 4 * base_chns, kernel_size=1, depth=depth, padding=0)
        self.conv2_1_mr = SingleResConv3D(2*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv2_2_mr = SingleResConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv2_3_mr = depthwise(8*base_chns, 4 * base_chns, kernel_size=1, depth=depth, padding=0)


        self.conv3_1_lr = SingleConv3D(4*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv3_2_lr = SingleConv3D(4 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv3_3_lr = depthwise(8 * base_chns, 4 * base_chns, kernel_size=1, depth=depth, padding=0)
        self.conv3_1_mr = SingleResConv3D(4*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv3_2_mr = SingleResConv3D(4 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv3_3_mr = depthwise(8*base_chns, 4 * base_chns, kernel_size=1, depth=depth, padding=0)

        self.conv4_1_lr = SingleConv3D(4*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=math.ceil(dilation/2), pad='same')
        self.conv4_2_lr = SingleConv3D(8 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=math.ceil(dilation/2), pad='same')
        self.conv4_3_lr = depthwise(16 * base_chns, 8 * base_chns, kernel_size=1, depth=depth, padding=0)
        self.conv4_1_mr = SingleResConv3D(4*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=math.ceil(dilation/2), pad='same')
        self.conv4_2_mr = SingleResConv3D(8 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=math.ceil(dilation/2), pad='same')
        self.conv4_3_mr = depthwise(16 * base_chns, 8 * base_chns, kernel_size=1, depth=depth, padding=0)


        self.conv5_1_lr = SingleConv3D(12*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv5_2_lr = SingleConv3D(8 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv5_3_lr = depthwise(16 * base_chns, 8 * base_chns, kernel_size=1, depth=depth, padding=0)
        self.conv5_1_mr = SingleResConv3D(12*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv5_2_mr = SingleResConv3D(8 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv5_3_mr = depthwise(16 * base_chns, 8 * base_chns, kernel_size=1, depth=depth, padding=0)

        self.conv6_1_lr = SingleConv3D(12*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv6_2_lr = SingleConv3D(4 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv6_3_lr = depthwise(8 * base_chns, 4 * base_chns, kernel_size=1, depth=depth, padding=0)
        self.conv6_1_mr = SingleResConv3D(12*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv6_2_mr = SingleResConv3D(4 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv6_3_mr = depthwise(8 * base_chns, 4 * base_chns, kernel_size=1, depth=depth, padding=0)

        self.conv7_1_lr = SingleConv3D(6*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv7_2_lr = SingleConv3D(2 * base_chns, 2 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv7_3_lr = depthwise(4 * base_chns, 2 * base_chns, kernel_size=1, depth=depth, padding=0)
        self.conv7_1_mr = SingleResConv3D(6*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv7_2_mr = SingleResConv3D(2 * base_chns, 2 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv7_3_mr = depthwise(4 * base_chns, 2 * base_chns, kernel_size=1, depth=depth, padding=0)

        self.classification_lr = nn.Sequential(
            nn.Conv3d(in_channels=2*base_chns, out_channels=n_classes, kernel_size=1),
        )
        self.classification_mr = nn.Sequential(
            nn.Conv3d(in_channels=2*base_chns, out_channels=n_classes, kernel_size=1),
        )


    def forward(self, x_mr, x_lr):
        out_lr = self.conv1_1_lr(x_lr)
        out_lr = self.conv1_2_lr(out_lr)
        out_mr = self.conv1_1_mr(x_mr)
        out_mr = self.conv1_2_mr(out_mr)
        conv1_lr = self.conv1_3_lr(t.cat((self.downinter_2(out_mr), out_lr), 1))
        conv1_mr = self.conv1_3_mr(t.cat((out_mr, self.upinter_2(out_lr)),1))
        out_lr = self.downsample(conv1_lr)  # 1/2
        out_mr = self.downsample(conv1_mr)  # 1/2

        out_lr = self.conv2_1_lr(out_lr)
        out_lr = self.conv2_2_lr(out_lr)
        out_mr = self.conv2_1_mr(out_mr)
        out_mr = self.conv2_2_mr(out_mr)
        conv2_lr = self.conv2_3_lr(t.cat((self.downinter_2(out_mr), out_lr), 1))
        conv2_mr = self.conv2_3_mr(t.cat((out_mr, self.upinter_2(out_lr)),1))
        out_lr = self.downsample(conv2_lr)  # 1/2
        out_mr = self.downsample(conv2_mr)  # 1/4

        out_lr = self.conv3_1_lr(out_lr)
        out_lr = self.conv3_2_lr(out_lr)
        out_mr = self.conv3_1_mr(out_mr)
        out_mr = self.conv3_2_mr(out_mr)
        conv3_lr = self.conv3_3_lr(t.cat((self.downinter_2(out_mr), out_lr), 1))
        conv3_mr = self.conv3_3_mr(t.cat((out_mr, self.upinter_2(out_lr)),1))
        out_lr = self.downsample(conv3_lr)  # 1/2
        out_mr = self.downsample(conv3_mr)  # 1/8

        out_lr = self.conv4_1_lr(out_lr)
        out_lr = self.conv4_2_lr(out_lr)
        out_mr = self.conv4_1_mr(out_mr)
        out_mr = self.conv4_2_mr(out_mr)
        out_lr = self.conv4_3_lr(t.cat((self.downinter_2(out_mr), out_lr), 1))
        out_mr = self.conv4_3_mr(t.cat((out_mr, self.upinter_2(out_lr)), 1))

        out_lr = self.drop(out_lr)  # 1/2
        out_mr = self.drop(out_mr)  # 1/2

        out_lr = self.upsample(out_lr)
        out_lr = t.cat((out_lr, conv3_lr), 1)
        out_lr = self.conv5_1_lr(out_lr)
        out_lr = self.conv5_2_lr(out_lr)
        out_mr = self.upsample(out_mr)
        out_mr = t.cat((out_mr, conv3_mr), 1)
        out_mr = self.conv5_1_mr(out_mr)
        out_mr = self.conv5_2_mr(out_mr)
        out_lr = self.conv5_3_lr(t.cat((self.downinter_2(out_mr), out_lr), 1))
        out_mr = self.conv5_3_mr(t.cat((out_mr, self.upinter_2(out_lr)), 1))

        out_lr = self.upsample(out_lr)
        out_lr = t.cat((out_lr, conv2_lr), 1)
        out_lr = self.conv6_1_lr(out_lr)
        out_lr = self.conv6_2_lr(out_lr)
        out_mr = self.upsample(out_mr)
        out_mr = t.cat((out_mr, conv2_mr), 1)
        out_mr = self.conv6_1_mr(out_mr)
        out_mr = self.conv6_2_mr(out_mr)
        out_lr = self.conv6_3_lr(t.cat((self.downinter_2(out_mr), out_lr), 1))
        out_mr = self.conv6_3_mr(t.cat((out_mr, self.upinter_2(out_lr)), 1))

        out_lr = self.upsample(out_lr)
        out_lr = t.cat((out_lr, conv1_lr), 1)
        out_lr = self.conv7_1_lr(out_lr)
        out_lr = self.conv7_2_lr(out_lr)
        out_mr = self.upsample(out_mr)
        out_mr = t.cat((out_mr, conv1_mr), 1)
        out_mr = self.conv7_1_mr(out_mr)
        out_mr = self.conv7_2_mr(out_mr)
        out_lr = self.conv7_3_lr(t.cat((self.downinter_2(out_mr), out_lr), 1))
        out_mr = self.conv7_3_mr(t.cat((out_mr, self.upinter_2(out_lr)), 1))

        out_lr = self.classification_lr(out_lr)
        out_mr = self.classification_mr(out_mr)
        predic = {'predic_lr':F.softmax(out_lr, dim=1),'predic_mr':F.softmax(out_mr, dim=1)}
        return predic
