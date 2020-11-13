from __future__ import print_function
from networks.module import Module
import torch
import torch.nn as nn
from networks.layers import TriResSeparateConv3D
import torch.nn.functional as F
import numpy as np

class Unet_adapt_Separate(Module):
    """
    相比普通U-net加入了res连接,并分离了3D卷积为3*3*1+1*1*3,最终几层宽度增加
    """
    def __init__(self, inc=1, n_classes=5, base_chns=12, droprate=0, norm='in', depth = False, dilation=1, separate_direction='axial', hu_lis=[0], norm_lis=[0]):
        super(Unet_adapt_Separate, self).__init__()
        self.model_name = "seg"
        self.base_hu = nn.Parameter(torch.tensor([-500], dtype=torch.float32))
        self.hu_lis = nn.Parameter(torch.FloatTensor(hu_lis))
        self.base_norm = nn.Parameter(torch.tensor([0], dtype=torch.float32))
        self.norm_lis = nn.Parameter(torch.FloatTensor(norm_lis))
        self.dropout = nn.Dropout(droprate)
        self.downsample = nn.MaxPool3d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')

        self.conv1_0 = TriResSeparateConv3D(1, base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv1_1 = TriResSeparateConv3D(1, base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv1_2 = TriResSeparateConv3D(1, base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv1_3 = TriResSeparateConv3D(1, base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv1_4 = TriResSeparateConv3D(1, base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)

        self.conv2 = TriResSeparateConv3D(5*base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)

        self.conv3 = TriResSeparateConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)


        self.conv4 = TriResSeparateConv3D(4 * base_chns, 8 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv5 = TriResSeparateConv3D(8 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)


        self.conv6_1 = TriResSeparateConv3D(8 * base_chns, 4 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv6_2 = TriResSeparateConv3D(4 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)

        self.conv7_1 = TriResSeparateConv3D(4 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv7_2 = TriResSeparateConv3D(2 * base_chns, 2 * base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)

        self.conv8_1 = TriResSeparateConv3D(7 * base_chns, 2*base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)
        self.conv8_2 = TriResSeparateConv3D(2*base_chns, 2*base_chns, norm=norm, depth=depth, pad='same', dilat=dilation, separate_direction=separate_direction)

        self.classification = nn.Sequential(
            nn.ReLU(),
            nn.Dropout3d(p=0.1),
            nn.Conv3d(in_channels=2*base_chns, out_channels=n_classes, kernel_size=1),
        )



    def forward(self, x):
        # self.hu_lis= nn.Parameter(torch.sort(self.hu_lis)[0])
        # self.norm_lis= nn.Parameter(torch.sort(self.norm_lis)[0])
        img = self.img_multi_thresh_normalized(x)
        out_0 = self.conv1_0(img[:,0:1])
        out_1 = self.conv1_1(img[:,1:2])
        out_2 = self.conv1_2(img[:,2:3])
        out_3 = self.conv1_3(img[:,3:4])
        out_4 = self.conv1_4(img[:,4::])
        conv1 = torch.cat((out_0,out_1,out_2,out_3,out_4),1)
        out = self.downsample(conv1)  # 1/2
        conv2 = self.conv2(out)  #
        out = self.downsample(conv2)  # 1/4
        conv3 = self.conv3(out)  #
        out = self.downsample(conv3)  # 1/8
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.dropout(out)

        out = self.upsample(out)  # 1/4
        out = torch.cat((out, conv3), 1)
        out = self.conv6_1(out)
        out = self.conv6_2(out)

        out = self.upsample(out)    # 1/2
        out = torch.cat((out, conv2), 1)
        out = self.conv7_1(out)
        out = self.conv7_2(out)

        out = self.upsample(out)  # 1/2
        out = torch.cat((out, conv1), 1)
        out = self.conv8_1(out)
        out = self.conv8_2(out)

        out = self.classification(out)
        predic = F.softmax(out, dim=1)
        return predic

    def img_multi_thresh_normalized(self, img):
        """
        :param img: tensor
        :param upthresh:
        :param downthresh:
        :param norm: norm or not
        :return:
        """
        for j in range(self.hu_lis.shape[0]): # 对所有transform类遍历
            cur_file = torch.zeros_like(img)
            for i in range(1, self.hu_lis.shape[1]): # 对一类中所有hu遍历
                hu_high = torch.sum(self.hu_lis[j,0:i+1]**2)
                hu_low = torch.sum(self.hu_lis[j,0:i]**2)
                norm_high = torch.sum(self.norm_lis[j,0:i+1]**2)
                norm_low = torch.sum(self.norm_lis[j,0:i]**2)
                mask = torch.where((img<(self.base_hu+hu_high))&(img>=(self.base_hu+hu_low)))
                k = (norm_high-norm_low)/(hu_high-hu_low)
                cur_file[mask] = k*(img[mask]-hu_low)+self.base_norm
            cur_file[torch.where(img >= (self.base_hu+hu_high))] = norm_high+self.base_norm
            if j==0:
                new_file = cur_file
            else:
                new_file = torch.cat((new_file, cur_file), dim=1)
        return new_file