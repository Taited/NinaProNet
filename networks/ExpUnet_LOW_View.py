from __future__ import print_function
from .module import Module
import torch as t
import torch.nn as nn
from .layers import Inception_v1_3D, Inception_v2_3D, DoubleResConv3D, Deconv3D, SingleConv3D, Res_concv_block
import torch.nn.functional as F
import math
class ExpUnetLOW(Module):
    """
    相比U-net少了一个下采样层
    """
    def __init__(self, n_classes=5, base_chns=48, droprate=0, depth=False):
        super(ExpUnetLOW, self).__init__()
        self.base_chns = base_chns
        self.conv1 = DoubleResConv3D(1, base_chns)  # (h，h)
        self.downsample1 = Inception_v1_3D(base_chns, base_chns, depth=depth)  # 1/2(h，h)
        self.conv2 = DoubleResConv3D(base_chns, 2*base_chns)  # 1/2(h，h)
        self.downsample2 = Inception_v1_3D(2*base_chns, 2*base_chns, depth=depth)  # 1/4(h，h)
        self.conv3 = DoubleResConv3D(2*base_chns, 4*base_chns, droprate=droprate)  # 1/4(h，h)
        self.downsample3 = Inception_v1_3D(4*base_chns, 4*base_chns, depth=depth)  # 1/8(h，h)
        self.conv4 = DoubleResConv3D(4*base_chns, 8*base_chns, droprate=droprate)  # 1/8(h，h)

        self.conv5 = SingleConv3D(8*base_chns, 4*base_chns, depth=depth)
        self.incept5 = Inception_v2_3D(4*base_chns, 4*base_chns, depth=depth)
        self.deconv5 = Deconv3D(4*base_chns, 4*base_chns)

        self.conv6 = SingleConv3D(8*base_chns, 2*base_chns, depth=depth)
        self.incept6 = Inception_v2_3D(2*base_chns, 2*base_chns, depth=depth)
        self.deconv6 = Deconv3D(2*base_chns, 2*base_chns)
        self.superDeconv6 = Deconv3D(4*base_chns, 4*base_chns)

        self.conv7 = SingleConv3D(4*base_chns, base_chns, depth=depth)
        self.incept7 = Inception_v2_3D(base_chns, base_chns, depth=depth)
        self.deconv7 = Deconv3D(base_chns, base_chns)
        self.superDeconv7 = Deconv3D(6*base_chns, 6*base_chns)

        self.conv8 = SingleConv3D(2*base_chns, base_chns, depth=depth)
        self.incept8 = Inception_v2_3D(base_chns, base_chns, depth=depth)
        self.superDeconv8 = Deconv3D(7*base_chns, 7*base_chns)

        self.classification = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv3d(in_channels=8*base_chns, out_channels=n_classes, kernel_size=1),
        )

    def _initialize_weights(self):
        '''
        the initialization
        '''

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_params(self, params):
        self.base_chns = params['base_feature_number']
        self.compres_chns = params['compress_feature_number']
        self.with_bn = params['with_bn']
        self.with_concatenate = params['with_concatenate']
        self.slice_margin = params.get('slice_margin', 0)

    def forward(self, x, droprate=0.2):
        conv1 = self.conv1(x)  # (512，512)
        down1 = self.downsample1(conv1)  # (256,256)
        conv2 = self.conv2(down1)  # (256,256)
        down2 = self.downsample2(conv2)  # (128,128)
        conv3 = self.conv3(down2)  # (128,128)
        down3 = self.downsample3(conv3)  # (64,64)
        conv4 = self.conv4(down3)  # (64,64)

        conv5 = self.incept5(self.conv5(conv4))  # 加入supervision
        supervision5 = conv5
        up5 = self.deconv5(conv5)
        up5 = t.cat((up5, conv3), 1)

        conv6 = self.incept6(self.conv6(up5))   # 加入supervision
        supervision6 = t.cat((self.superDeconv6(supervision5), conv6), 1)
        up6 = self.deconv6(conv6)
        up6 = t.cat((up6, conv2), 1)

        conv7 = self.incept7(self.conv7(up6))   # 加入supervision
        supervision7 = t.cat((self.superDeconv7(supervision6), conv7), 1)
        up7 = self.deconv7(conv7)
        up7 = t.cat((up7, conv1), 1)

        conv8 = self.incept8(self.conv8(up7))
        supervision8 = t.cat((self.superDeconv8(supervision7), conv8), 1)

        cls = self.classification(supervision8)
        predic = F.softmax(cls, dim=1)
        return predic

class TensorSliceLayer(nn.Module):
    """
    extract the central part of a tensor
    """

    def __init__(self, margin=1, regularizer=None, name='tensor_extract'):
        self.layer_name = name
        super(TensorSliceLayer, self).__init__(name=self.layer_name)
        self.margin = margin

    def layer_op(self, input_tensor):
        output_tensor = input_tensor[:, self.margin: -self.margin]
        return output_tensor