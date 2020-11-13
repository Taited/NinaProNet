import numpy as np
from torch.autograd import Variable
import torch.nn as nn
# coding:utf8
import torch

class AttentionExpDiceLoss(nn.Module):
    def __init__(self, n_class, alpha, gama=0.0001):
        super(AttentionExpDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class
        self.gama = gama
        self.weight = [1,1]
        self.alpha = alpha
        smooth = 1
        self.Ldice = Ldice(n_class, smooth)
        self.Lcross = Lcross(n_class)
    def forward(self, input, label):
        '''
        :param input: batch*class*depth*length*height or batch*calss*length*height
        :param label: batch*depth*length*height or batch*length*height  注意未经过one_hot，直接输入label即可
        :param dis: batch*class*depth*length*height or batch*calss*length*height
        :return:
        '''
        smooth = 1
        batch_size = input.size(0)
        realinput = input
        reallabel = label
        input = input.view(batch_size, self.n_class, -1)
        label = self.one_hot_encoder(label).contiguous().view(batch_size, self.n_class, -1)
        attentionseg = torch.exp((input - label)/self.alpha) * input
        label_sum = torch.sum(label[:, 1::], 2) + smooth  # 非背景类label各自和
        Wl = (torch.sum(label_sum) / torch.sum(label_sum, 0))**0.5  # 各label占总非背景类label比值的开方
        Ldice = self.Ldice(attentionseg, label, batch_size)   #
        Lcross = self.Lcross(realinput, reallabel, Wl, label_sum)
        Lexp = self.weight[0] * Ldice + self.weight[1] * Lcross
        return Lexp

class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.eye(depth).cuda()  # torch.sparse.torch.eye
                                             # eye生成depth尺度的单位矩阵

    def forward(self, X_in):
        '''
        :param X_in: batch*depth*length*height or batch*length*height
        :return: batch*class*depth*length*height or batch*calss*length*height
        '''
        n_dim = X_in.dim()  # 返回dimension数目
        output_size = X_in.size() + torch.Size([self.depth])   # 增加一个class通道
        num_element = X_in.numel()  # 返回element总数
        X_in = X_in.data.long().view(num_element)   # 将label拉伸为一行
        out1 = Variable(self.ones.index_select(0, X_in))
        out = out1.view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()  # permute更改dimension顺序

class Ldice(nn.Module):
    def __init__(self, smooth, n_class):
        super(Ldice, self).__init__()
        self.smooth = smooth
        self.n_class = n_class

    def forward(self, input, label, batch_size):
        '''
        Ldice
        '''
        inter = torch.sum(input * label, 2) + self.smooth
        union1 = torch.sum(input, 2) + self.smooth
        union2 = torch.sum(label, 2) + self.smooth
        dice = 2.0 * inter / (union1 + union2)
        logdice = -torch.log(dice)
        expdice = torch.sum(logdice) # ** self.gama
        Ldice = expdice / (batch_size*self.n_class)
        return Ldice

class Lcross(nn.Module):
    def __init__(self, n_class):
        super(Lcross, self).__init__()
        self.n_class = n_class
    def forward(self, realinput, reallabel, Wl, label_sum):
        '''
        realinput:
        reallabel:
        Wl: 各label占总非背景类label比值的开方
        '''
        Lcross = 0
        for i in range(1, self.n_class):
            mask = reallabel == i
            if torch.sum(mask).item() > 0:
                ProLabel = realinput[:, i][mask.detach()]
                LogLabel = -torch.log(ProLabel)
                ExpLabel = torch.sum(LogLabel)+1  # **self.gama
                Lcross += Wl[i - 1] * ExpLabel
        Lcross = Lcross / torch.sum(label_sum)

        return Lcross
