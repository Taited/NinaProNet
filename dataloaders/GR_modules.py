import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import cv2

# conv_block(nn.Module) for U-net convolution block
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, drop_out=False):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        self.dropout = drop_out

    def forward(self, x):
        x = self.conv(x)
        if self.dropout:
            x = nn.Dropout2d(0.5)(x)
        return x


# # UpCat(nn.Module) for U-net UP convolution
class UpCat(nn.Module):
    def __init__(self, in_feat, out_feat, is_deconv=True):
        super(UpCat, self).__init__()

        if is_deconv:
            self.up = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
        else:
            # self.conv = conv_block(in_feat + out_feat, out_feat)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        # outputs = self.up(inputs)
        outputs = self.up(down_outputs)
        offset = inputs.size()[3] - outputs.size()[3]
        if offset == 1:
            addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2]), out=None).unsqueeze(3).cuda()
            outputs = torch.cat([outputs, addition], dim=3)
        elif offset > 1:
            addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2], offset), out=None).cuda()
            outputs = torch.cat([outputs, addition], dim=3)
        out = torch.cat([inputs, outputs], dim=1)

        return out


# # UpCatconv(nn.Module) for Atten U-net UP convolution
class UpCatconv(nn.Module):
    def __init__(self, in_feat, out_feat, is_deconv=True, drop_out=False):
        super(UpCatconv, self).__init__()

        if is_deconv:
            self.conv = conv_block(in_feat, out_feat, drop_out=drop_out)
            self.up = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
        else:
            self.conv = conv_block(in_feat + out_feat, out_feat, drop_out=drop_out)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        # outputs = self.up(inputs)
        outputs = self.up(down_outputs)
        offset = inputs.size()[3] - outputs.size()[3]
        if offset == 1:
            addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2]), out=None).unsqueeze(3).cuda()
            outputs = torch.cat([outputs, addition], dim=3)
        elif offset > 1:
            addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2], offset), out=None).cuda()
            outputs = torch.cat([outputs, addition], dim=3)
        out = self.conv(torch.cat([inputs, outputs], dim=1))

        return out


# # UnetGridGatingSignal3(nn.Module)
class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1, 1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, (1, 1), (0, 0)),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, (1, 1), (0, 0)),
                                       nn.ReLU(inplace=True),
                                       )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


# # GridAttentionBlock2D(nn.Module)
class GridAttentionBlock2D(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None):
        super(GridAttentionBlock2D, self).__init__()

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # Output transform
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=(1, 1), stride=1, padding=0, bias=True)
        self.phi = nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels,
                             kernel_size=(1, 1), stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1,
                             padding=0, bias=True)

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode='bilinear')
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode='bilinear')
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(size=scale_factor, mode='bilinear'), )

    def forward(self, input):
        return self.dsv(input)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


# # SE block add to U-net
def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class SE_Conv_Block_fetus(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(SE_Conv_Block_fetus, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        if planes == 16:
            self.globalAvgPool = nn.AvgPool2d((256, 256), stride=1)  # (224, 300) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((256, 256), stride=1)
        elif planes == 32:
            self.globalAvgPool = nn.AvgPool2d((128, 128), stride=1)  # (112, 150) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((128, 128), stride=1)
        elif planes == 64:
            self.globalAvgPool = nn.AvgPool2d((64, 64), stride=1)    # (56, 75) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((64, 64), stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d((32, 32), stride=1)    # (28, 37) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((32, 32), stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d((16, 16), stride=1)    # (14, 18) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((16, 16), stride=1)

        self.fc1 = nn.Linear(in_features=planes * 2, out_features=round(planes / 2))
        self.fc2 = nn.Linear(in_features=round(planes / 2), out_features=planes * 2)
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)
        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        out1 = out
        # For global average pool
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out
        # For global maximum pool
        out1 = self.globalMaxPool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1)
        out1 = out1 * original_out

        out += out1
        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out


class SE_Conv_Block_nomax_fetus(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(SE_Conv_Block_nomax_fetus, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        if planes == 16:
            self.globalAvgPool = nn.AvgPool2d((256, 256), stride=1)  # (224, 300) for ISIC2018
            # self.globalMaxPool = nn.MaxPool2d((224, 300), stride=1)
        elif planes == 32:
            self.globalAvgPool = nn.AvgPool2d((128, 128), stride=1)  # (112, 150) for ISIC2018
            # self.globalMaxPool = nn.MaxPool2d((112, 150), stride=1)
        elif planes == 64:
            self.globalAvgPool = nn.AvgPool2d((64, 64), stride=1)    # (56, 75) for ISIC2018
            # self.globalMaxPool = nn.MaxPool2d((56, 75), stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d((32, 32), stride=1)    # (28, 37) for ISIC2018
            # self.globalMaxPool = nn.MaxPool2d((28, 37), stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d((16, 16), stride=1)    # (14, 18) for ISIC2018
            # self.globalMaxPool = nn.MaxPool2d((14, 18), stride=1)

        self.fc1 = nn.Linear(in_features=planes * 2, out_features=round(planes / 2))
        self.fc2 = nn.Linear(in_features=round(planes / 2), out_features=planes * 2)
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)
        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        # For global average pool
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out

        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out


class SE_Conv_Block_isic(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(SE_Conv_Block_isic, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        if planes <= 16:
            self.globalAvgPool = nn.AvgPool2d((224, 300), stride=1)  # (224, 300) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((224, 300), stride=1)
        elif planes == 32:
            self.globalAvgPool = nn.AvgPool2d((112, 150), stride=1)  # (112, 150) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((112, 150), stride=1)
        elif planes == 64:
            self.globalAvgPool = nn.AvgPool2d((56, 75), stride=1)    # (56, 75) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((56, 75), stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d((28, 37), stride=1)    # (28, 37) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((28, 37), stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d((14, 18), stride=1)    # (14, 18) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((14, 18), stride=1)

        self.fc1 = nn.Linear(in_features=planes * 2, out_features=round(planes / 2))
        self.fc2 = nn.Linear(in_features=round(planes / 2), out_features=planes * 2)
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)
        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        out1 = out
        # For global average pool
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out
        # For global maximum pool
        out1 = self.globalMaxPool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1)
        out1 = out1 * original_out

        out += out1
        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out


class SE_Conv_Block_nomax_isic(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(SE_Conv_Block_nomax_isic, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        if planes == 16:
            self.globalAvgPool = nn.AvgPool2d((224, 300), stride=1)  # (224, 300) for ISIC2018
            # self.globalMaxPool = nn.MaxPool2d((224, 300), stride=1)
        elif planes == 32:
            self.globalAvgPool = nn.AvgPool2d((112, 150), stride=1)  # (112, 150) for ISIC2018
            # self.globalMaxPool = nn.MaxPool2d((112, 150), stride=1)
        elif planes == 64:
            self.globalAvgPool = nn.AvgPool2d((56, 75), stride=1)    # (56, 75) for ISIC2018
            # self.globalMaxPool = nn.MaxPool2d((56, 75), stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d((28, 37), stride=1)    # (28, 37) for ISIC2018
            # self.globalMaxPool = nn.MaxPool2d((28, 37), stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d((14, 18), stride=1)    # (14, 18) for ISIC2018
            # self.globalMaxPool = nn.MaxPool2d((14, 18), stride=1)

        self.fc1 = nn.Linear(in_features=planes * 2, out_features=round(planes / 2))
        self.fc2 = nn.Linear(in_features=round(planes / 2), out_features=planes * 2)
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)
        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        # For global average pool
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out

        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out


# # Scale attention block add to U-net
class SC_Conv_Block(nn.Module):
    def __init__(self, in_size, out_size1, out_size2, padding_list, dilation_list, num_branches, drop_out=False):
        super(SC_Conv_Block, self).__init__()
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.num_branches = num_branches
        self.dropout = drop_out

        self.conv1 = nn.Conv2d(in_size, out_size1, kernel_size=3, dilation=2, padding=2)
        self.bn1 = nn.BatchNorm2d(out_size1)
        self.bn_list1 = nn.ModuleList()
        for i in range(len(self.padding_list)):
            self.bn_list1.append(nn.BatchNorm2d(out_size2))

        self.conv2 = nn.Conv2d(out_size1, out_size2, kernel_size=3,
                               dilation=self.dilation_list[0], padding=self.padding_list[0])
        # self.bn2 = nn.BatchNorm2d(out_size2)
        self.convatt1 = nn.Conv2d(out_size1, int(out_size1 / 2), kernel_size=3, padding=1)
        self.convatt2 = nn.Conv2d(int(out_size1 / 2), self.num_branches, kernel_size=1)

        self.conv3 = conv3x3(out_size2, out_size2)
        self.bn3 = nn.BatchNorm2d(out_size2)

        self.relu = nn.ReLU(inplace=True)
        if in_size == out_size2:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_size, out_size2, kernel_size=1), nn.BatchNorm2d(out_size2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # compute attention weights in the first autofocus convolutional layer
        feature = x.detach()
        att = self.relu(self.convatt1(feature))
        att = self.convatt2(att)
        att = F.softmax(att, dim=1)

        # linear combination of different rates
        x1 = self.conv2(x)
        shape = x1.size()
        x1 = self.bn_list1[0](x1) * att[:, 0:1, :, :].expand(shape)

        # sharing weights in parallel convolutions
        for i in range(1, self.num_branches):
            x2 = F.conv2d(x, self.conv2.weight, padding=self.padding_list[i], dilation=self.dilation_list[i])
            x2 = self.bn_list1[i](x2)
            x1 += x2 * att[:, i:(i + 1), :, :].expand(shape)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x1 + residual
        x = self.relu(x)

        out = self.conv3(x)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out


# # Scale attention block add to U-net
class Cat_SC_Block(nn.Module):
    def __init__(self, in_size, out_size1, out_size2, drop_out=False):
        super(Cat_SC_Block, self).__init__()
        self.dropout = drop_out
        self.num_branches = out_size2

        self.conv1 = nn.Conv2d(in_size, out_size1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_size1)
        self.bn_list1 = nn.ModuleList()
        for i in range(out_size2):
            self.bn_list1.append(nn.BatchNorm2d(out_size2))

        self. conv2 = nn.Conv2d(in_size, out_size2, kernel_size=3,
                                stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size2)
        self.convatt1 = nn.Conv2d(in_size, int(out_size1 / 2), kernel_size=3, padding=1)
        self.convatt2 = nn.Conv2d(int(out_size1 / 2), out_size2, kernel_size=1)

        self.conv3 = conv3x3(out_size1, out_size2)
        self.bn3 = nn.BatchNorm2d(out_size2)

        self.relu = nn.ReLU(inplace=True)
        if in_size == out_size2:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_size, out_size2, kernel_size=1), nn.BatchNorm2d(out_size2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x

        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # compute attention weights in the first autofocus convolutional layer
        feature = x.detach()
        att = self.relu(self.convatt1(feature))
        att = self.convatt2(att)
        att = F.softmax(att, dim=1)

        # linear combination of different rates
        x1 = self.conv2(x)
        shape = x1.size()
        x1 = self.bn_list1[0](x1) * att[:, 0:1, :, :].expand(shape)

        # sharing weights in parallel convolutions
        for i in range(1, self.num_branches):
            x2 = F.conv2d(x, self.conv2.weight, padding=1)
            x2 = self.bn_list1[i](x2)
            x1 += x2 * att[:, i:(i + 1), :, :].expand(shape)

        if self.downsample is not None:
            residual = self.downsample(residual)

        # print('x1:{}, residual:{}'.format(x1.size(), residual.size()))
        x = x1 + residual
        out = self.relu(x)

        # out = self.conv3(x)
        # out = self.bn3(out)
        # out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out


# # CBAM Convolutional block attention module
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # scalecoe = F.sigmoid(channel_att_sum)
        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale, scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)    # broadcasting
        # spa_scale = scale.expand_as(x)
        # print(spa_scale.shape)
        return x * scale, scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out, atten = self.ChannelGate(x)
        if not self.no_spatial:
            x_out, spatial_att = self.SpatialGate(x_out)
        return x_out, spatial_att


class CBAM_conv_block(nn.Module):
    def __init__(self, in_size, out_size, stride=1, downsample=None, use_cbam=True, no_spatial=False, drop_out=False):
        super(CBAM_conv_block, self).__init__()
        if stride != 1 or in_size != out_size:
            downsample = nn.Sequential(
                nn.Conv2d(in_size, out_size,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_size),
            )
        self.downsample = downsample
        self.stride = stride
        self.no_spatial = no_spatial
        self.dropout = drop_out

        self.conv1 = conv3x3(in_size, out_size, self.stride)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_size, out_size)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.conv3 = conv3x3(out_size, out_size)
        self.bn3 = nn.BatchNorm2d(out_size)  # nn.GroupNorm(4, out_size)

        if use_cbam:
            self.cbam = CBAM(out_size, reduction_ratio=4, no_spatial=self.no_spatial)  # out_size
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out, _ = self.cbam(out)

        out += residual
        out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)
        # out = self.relu(out)

        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out


class SCAL_conv_block(nn.Module):
    def __init__(self, in_size, out_size, stride=1, downsample=None, use_cbam=True, no_spatial=False, drop_out=False):
        super(SCAL_conv_block, self).__init__()
        # if stride != 1 or in_size != out_size:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(in_size, out_size,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.GroupNorm(4, out_size),
        #     )
        self.downsample = downsample
        self.stride = stride
        self.no_spatial = no_spatial
        self.dropout = drop_out

        # self.conv1 = conv3x3(in_size, out_size, self.stride)
        # self.bn1 = nn.GroupNorm(4, out_size)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(out_size, out_size)
        # self.bn2 = nn.GroupNorm(4, out_size)
        self.conv3 = conv3x3(in_size, out_size)
        self.bn3 = nn.GroupNorm(4, out_size)  # nn.GroupNorm(4, out_size)

        if use_cbam:
            self.cbam = CBAM(in_size, reduction_ratio=4, no_spatial=self.no_spatial)  # out_size
        else:
            self.cbam = None

    def forward(self, x):
        residual = x
        # out_arr = x.data.cpu().numpy()
        # for i in range(out_arr.shape[0]):
        #     for j in range(out_arr.shape[1]):
        #         print(out_arr[i, j, :, :].min(), out_arr[i, j, :, :].max())
        #         print(out_arr[i, j, :, :])
        #         cv2.imwrite('./picture/ISIC_' + str(i) + '_' + str(j) + '.png', out_arr[i, j, :, :] * 255)
        out = x

        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        #
        # out = self.conv2(out)
        # out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out, atten_map = self.cbam(out)
            out_att = atten_map.data.cpu().numpy()
            # print(atten_map.shape)
        # out_arr = self.relu(out).data.cpu().numpy()
            for i in range(out_att.shape[0]):
                for j in range(out_att.shape[1]):
                    print(out_att[i, j, :, :].min(), out_att[i, j, :, :].max())
                    print(out_att[i, j, :, :])
                    cv2.imwrite('./picture/ISIC_' + str(i) + '_' + str(j) + '.png', out_att[i, j, :, :] * 255)

        out += residual

        # Output for adding residual
        # out_arr = self.relu(out).data.cpu().numpy()
        # out_fusion = np.multiply(out_arr[0, 6, :, :], out_arr[0, 14, :, :])
        # out_arr = np.max(out_arr[i], 0)
        # print(out_arr[0].shape, out_arr[0].max())
        # for i in range(out_arr.shape[0]):
        #     for j in range(out_arr.shape[1]):
                # print(out_arr[i, j, :, :].max())
        # cv2.imwrite('./picture/ISIC_' + 'fusion_' + '.png', out_fusion * 255)
        # out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out
