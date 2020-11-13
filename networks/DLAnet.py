from __future__ import print_function
from .module import Module
import torch as t
import torch.nn as nn
from .layers import *
import torch.nn.functional as F

class DLAnet(Module):
    def __init__(self, n_classes=5, base_chns=32):
        super(DLAnet, self).__init__()
        self.base_chns = base_chns

        "第一层"
        self.BasicConv1 = DLAConv3D(1, base_chns)
        self.BasicConv2 = DLAConv3D(base_chns, 2*base_chns)
        self.BasicConv3 = DLAConv3D(2*base_chns, 4*base_chns)
        self.BasicConv4 = DLAConv3D(4*base_chns, 6*base_chns)
        self.BasicConv5 = DLAConv3D(6*base_chns, 8*base_chns)

        "第二层"
        self.SecondConv1 = AggreResNode3D(3*base_chns, 2*base_chns)
        self.SecondConv2 = AggreResNode3D(6*base_chns, 4*base_chns)
        self.SecondConv3 = AggreResNode3D(10*base_chns, 6*base_chns)
        self.SecondConv4 = AggreResNode3D(14*base_chns, 8*base_chns)

        "第三层"
        self.ThirdConv1 = AggreResNode3D(6*base_chns, 4*base_chns)
        self.ThirdConv2 = AggreResNode3D(10*base_chns, 6*base_chns)
        self.ThirdConv3 = AggreResNode3D(14*base_chns, 8*base_chns)

        "第四层"
        self.FourthConv1 = AggreResNode3D(10*base_chns, 6*base_chns)
        self.FourthConv2 = AggreResNode3D(14*base_chns, 8*base_chns)

        "输出层"
        self.FifhConv = AggreResNode3D(14*base_chns, 8*base_chns)
        self.classification = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv3d(8*base_chns, n_classes, 1)
        )

    def set_params(self, params):
        self.base_chns = params['base_feature_number']
        self.compres_chns = params['compress_feature_number']
        self.with_bn = params['with_bn']
        self.with_concatenate = params['with_concatenate']
        self.slice_margin = params.get('slice_margin', 0)

    def forward(self, x):
        BasicConv1 = self.BasicConv1(x)
        BasicConv2 = self.BasicConv2(BasicConv1)
        BasicConv3 = self.BasicConv3(BasicConv2)
        BasicConv4 = self.BasicConv4(BasicConv3)
        BasicConv5 = self.BasicConv5(BasicConv4)

        SecondConv1 = self.SecondConv1(BasicConv1, BasicConv2)
        SecondConv2 = self.SecondConv2(BasicConv2, BasicConv3)
        SecondConv3 = self.SecondConv3(BasicConv3, BasicConv4)
        SecondConv4 = self.SecondConv4(BasicConv4, BasicConv5)

        ThirdConv1 = self.ThirdConv1(SecondConv1, SecondConv2)
        ThirdConv2 = self.ThirdConv2(SecondConv2, SecondConv3)
        ThirdConv3 = self.ThirdConv3(SecondConv3, SecondConv4)

        FourthConv1 = self.FourthConv1(ThirdConv1, ThirdConv2)
        FourthConv2 = self.FourthConv2(ThirdConv2, ThirdConv3)

        FifthConv = self.FifhConv(FourthConv1, FourthConv2)

        cls = self.classification(FifthConv)

        return cls


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
