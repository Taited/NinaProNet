# coding:utf8
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch import eq, sum, gt   # eq返回相同元素索引,gt返回大于给定值索引
from torch.nn import init
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt


class TestDiceLoss(nn.Module):
    def __init__(self, n_class):
        super(TestDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class

    def forward(self, pred, label, show=False):
        smooth = 0.00001
        batch_size = pred.size(0)
        pred = torch.max(pred, 1)[1]
        pred = self.one_hot_encoder(pred).contiguous().view(batch_size, self.n_class, -1)
        label = self.one_hot_encoder(label).contiguous().view(batch_size, self.n_class, -1)
        inter = torch.sum(torch.sum(pred * label, 2), 0) + smooth
        union1 = torch.sum(torch.sum(pred, 2), 0) + smooth
        union2 = torch.sum(torch.sum(label, 2), 0) + smooth


        '''
        为避免当前训练图像中未出现的器官影响dice,删除dice大于0.98的部分
        '''
        andU = 2.0 * inter / (union1 + union2)
        score = andU

        return score.float()

class SoftDiceLoss(nn.Module):
    def __init__(self, n_class):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class

    def forward(self, pred, label):
        '''
        :param pred: the prediction, batchsize*n_class*depth*length*width
        :param label: the groundtruth, batchsize*depth*length*width
        :return: loss
        '''
        smooth = 1
        batch_size = pred.size(0)
        pred = pred.view(batch_size, self.n_class, -1)
        label = self.one_hot_encoder(label).contiguous().view(batch_size, self.n_class, -1)
        inter = torch.sum(pred * label, 2) + smooth
        union1 = torch.sum(pred, 2) + smooth
        union2 = torch.sum(label, 2) + smooth

        andU = torch.sum(2.0 * inter/(union1 + union2))
        score = 1 - andU/(batch_size*self.n_class)

        return score

class FocalLoss(nn.Module):
    def __init__(self, n_class):
        super(FocalLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class

    def forward(self, pred, label):
        '''
        :param pred: the prediction, batchsize*n_class*depth*length*width
        :param label: the groundtruth, batchsize*depth*length*width
        :return: loss
        '''
        batch_size = pred.size(0)
        pred = pred.view(batch_size, self.n_class, -1)
        label = self.one_hot_encoder(label).contiguous().view(batch_size, self.n_class, -1)
        volume = pred.size(2)
        score = -torch.sum(label*(1-pred)**2*torch.log(pred))/volume

        return score

class BinaryCrossEntropy(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()
        self.smooth=0.0001
    def forward(self, pred,label):
        '''
        :param pred: the prediction, batchsize*n_class*depth*length*width
        :param label: the groundtruth, batchsize*depth*length*width
        :return: loss
        '''
        batch_size = pred.size(0)
        pred = pred.view(batch_size, -1).float()
        label = label.view(batch_size, -1).float()
        volume = batch_size*pred.size(1)
        fgvolume = torch.sum(label)+1
        bgvolume = volume-fgvolume+1
        score = -torch.sum(torch.log(bgvolume/fgvolume)*label*torch.log(pred+self.smooth)+
                           (1-label)*torch.log(1-pred+self.smooth))/volume

        return score


class Focal_and_Dice_loss(nn.Module):
    def __init__(self, n_class, lamda=1):
        super(Focal_and_Dice_loss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class
        self.lamda = lamda
        self.FocalLoss = FocalLoss(n_class)
        self.SoftDiceloss = SoftDiceLoss(n_class)

    def forward(self, pred, label):
        '''
        :param pred: the prediction, batchsize*n_class*depth*length*width
        :param label: the groundtruth, batchsize*depth*length*width
        :return: loss
        '''
        score = self.lamda*self.FocalLoss(pred, label)+self.n_class*self.SoftDiceloss(pred, label)
        return score

class CrossEntropy_and_Dice_Loss(nn.Module):
    def __init__(self, n_class, lamda=1):
        super(CrossEntropy_and_Dice_Loss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class
        self.lamda = lamda
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.SoftDiceloss = SoftDiceLoss(n_class)

    def forward(self, pred, label):
        '''
        :param pred: the prediction, batchsize*n_class*depth*length*width
        :param label: the groundtruth, batchsize*depth*length*width
        :return: loss
        '''
        score = self.lamda*self.CrossEntropyLoss(pred, label)+self.n_class*self.SoftDiceloss(pred, label)
        return score

class BiaseDiceLoss(nn.Module):
    def __init__(self, n_class, alpha=1):
        super(BiaseDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class
        self.alpha = alpha
    def forward(self, pred, label):
        '''
        :param pred: the prediction, batchsize*n_class*depth*length*width
        :param label: the groundtruth, batchsize*depth*length*width
        :return: loss
        '''
        smooth = 0.01
        batch_size = pred.size(0)

        pred = pred.view(batch_size, self.n_class, -1)
        label = self.one_hot_encoder(label).contiguous().view(batch_size, self.n_class, -1)

        inter = torch.sum(pred * label, 2) + smooth
        union1 = torch.sum(pred*(1-label), 2) + smooth
        union2 = self.alpha*torch.sum(label*(1-pred), 2) + smooth

        andU = torch.sum(2.0 * inter/(union1 + union2+2*inter))
        score = 1 - andU/(batch_size*self.n_class)

        return score

class AttentionDiceLoss(nn.Module):
    def __init__(self, n_class, alpha):
        super(AttentionDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class
        self.alpha = alpha
    def forward(self, pred, label):
        '''
        :param pred: the prediction, batchsize*n_class*depth*length*width
        :param label: the groundtruth, batchsize*depth*length*width
        :return: loss
        '''
        smooth = 0.01
        batch_size = pred.size(0)

        pred = pred.view(batch_size, self.n_class, -1)
        label = self.one_hot_encoder(label).contiguous().view(batch_size, self.n_class, -1)
        attentioninput = torch.exp((pred - label) / self.alpha) * pred
        inter = torch.sum(attentioninput * label, 2) + smooth
        union1 = torch.sum(attentioninput, 2) + smooth
        union2 = torch.sum(label, 2) + smooth

        andU = torch.sum(2.0 * inter / (union1 + union2))
        score = batch_size * self.n_class - andU

        return score

class ExpDiceLoss(nn.Module):
    def __init__(self, n_class, weights=[1, 1], gama=0.0001):
        super(ExpDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class
        self.gama = gama
        self.weight = weights
        smooth = 1
        self.Ldice = Ldice(n_class, smooth)
        self.Lcross = Lcross(n_class)
    def forward(self, pred, label):
        '''
        :param pred: batch*class*depth*length*height or batch*calss*length*height
        :param label: batch*depth*length*height or batch*length*height
        :return:
        '''
        smooth = 1
        batch_size = pred.size(0)
        realinput = pred
        reallabel = label

        pred = pred.view(batch_size, self.n_class, -1)
        label = self.one_hot_encoder(label).contiguous().view(batch_size, self.n_class, -1)
        label_sum = torch.sum(label[:, 1::], 2) + smooth  # 非背景类label各自和
        Wl = (torch.sum(label_sum) / torch.sum(label_sum, 0))**0.5  # 各label占总非背景类label比值的开方
        Ldice = self.Ldice(pred, label, batch_size)   #
        Lcross = self.Lcross(realinput, reallabel, Wl, label_sum)
        Lexp = self.weight[0] * Ldice + self.weight[1] * Lcross
        return Lexp
class AttentionExpDiceLoss(nn.Module):
    def __init__(self, n_class, alpha, gama=0.0001, weight=[1,1]):
        super(AttentionExpDiceLoss, self).__init__()
        self.n_class = n_class
        self.gama = gama
        self.weight = weight
        self.alpha = alpha
        self.smooth = 1
        self.Ldice = Ldice(n_class, self.smooth)
        self.Lcross = Lcross(n_class)
    def forward(self, pred, label):
        '''
        :param pred: batch*class*depth*length*height or batch*calss*length*height
        :param label: batch*depth*length*height or batch*length*height
        :param dis: batch*class*depth*length*height or batch*calss*length*height
        :return:
        '''
        batch_size = pred.size(0)
        realinput = pred.clone()
        reallabel = label.clone()
        pred = pred.view(batch_size, self.n_class, -1)[:, 1::]
        label = get_soft_label(label, self.n_class).view(batch_size, self.n_class, -1)[:, 1::].float()
        att_pred = torch.exp((pred - label)/self.alpha) * pred
        label_sum = torch.sum(label, 2) + self.smooth  # 非背景类label各自和
        Wl = ((torch.sum(label_sum)+self.smooth) / (torch.sum(label_sum, 0)+self.smooth))**0.5  # 各label占总非背景类label比值的开方
        Ldice = self.Ldice(att_pred, label, batch_size)   #
        Lcross = self.Lcross(realinput, reallabel, Wl, label_sum)
        Lexp = self.weight[0] * Ldice + self.weight[1] * Lcross
        return Lexp
class Ldice(nn.Module):
    def __init__(self, smooth, n_class):
        super(Ldice, self).__init__()
        self.smooth = smooth
        self.n_class = n_class

    def forward(self, pred, label, batch_size):
        '''
        Ldice
        '''
        dice = 2.0 * (torch.sum(pred * label, 2) + self.smooth) / (torch.sum(pred, 2) + 2*self.smooth + torch.sum(label, 2))
        logdice = -torch.log(dice)
        expdice = torch.sum(logdice) # ** self.gama
        Ldice = expdice / (batch_size*self.n_class)
        return Ldice
class Lcross(nn.Module):
    def __init__(self, n_class, smooth=1):
        super(Lcross, self).__init__()
        self.n_class = n_class
        self.smooth = smooth
    def forward(self, pred, label, Wl, label_sum):
        '''
        pred:N*C*...
        label:N*...
        Wl: 各label占总非背景类label比值的开方
        '''
        Lcross = 0
        for i in range(1, self.n_class):
            mask = label == i
            if torch.sum(mask).item() > 0:
                ExpLabel = torch.sum(-torch.log(pred[:, i][mask.detach()]+self.smooth))
                Lcross += Wl[i - 1] * ExpLabel
        Lcross = Lcross / torch.sum(label_sum)

        return Lcross

class AttentionInteractExpDiceLoss(nn.Module):
    def __init__(self, n_class, weights=[1, 1], gama=0.0001):
        super(AttentionInteractExpDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class
        self.gama = gama
        self.weight = weights
        smooth = 1
        self.Ldice = Ldice(n_class, smooth)
        self.Lcross = Lcross(n_class)
    def forward(self, pred, label):
        '''
        :param pred: batch*class*depth*length*height or batch*calss*length*height
        :param label: batch*depth*length*height or batch*length*height
        :param dis: batch*class*depth*length*height or batch*calss*length*height
        :return:
        '''
        smooth = 1
        batch_size = pred.size(0)
        realinput = pred
        reallabel = label
        pred = pred.view(batch_size, self.n_class, -1)
        label = self.one_hot_encoder(label).contiguous().view(batch_size, self.n_class, -1)
        attentioninteractseg = torch.exp(pred-label)
        label_sum = torch.sum(label[:, 1::], 2) + smooth  # 非背景类label各自和
        Wl = (torch.sum(label_sum) / torch.sum(label_sum, 0))**0.5  # 各label占总非背景类label比值的开方
        Ldice = self.Ldice(attentioninteractseg, label, batch_size)   #
        Lcross = self.Lcross(realinput, reallabel, Wl, label_sum)
        Lexp = self.weight[0] * Ldice + self.weight[1] * Lcross
        return Lexp

class InteractExpLoss(nn.Module):
    def __init__(self, n_class, weights=[1, 1], gama=0.0001, alpha=1):
        super(InteractExpLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class
        self.gama = gama
        self.weight = weights
        smooth = 1
        self.Ldice = Ldice(n_class, smooth)
        self.InteractLcross = InteractLcross(n_class, alpha)

    def forward(self, pred, label, dis):
        '''
        :param pred: batch*class*depth*length*height or batch*calss*length*height
        :param label: batch*depth*length*height or batch*length*height
        :return:
        '''
        smooth = 1
        batch_size = pred.size(0)
        realinput = pred
        reallabel = label

        pred = pred.view(batch_size, self.n_class, -1)
        label = self.one_hot_encoder(label).contiguous().view(batch_size, self.n_class, -1)
        # dis = dis.contiguous().view(batch_size, self.n_class, -1)
        label_sum = torch.sum(label, 2) + smooth  # 各类自和
        Wl = (torch.sum(label_sum) / torch.sum(label_sum, 0)) ** 0.3  # 各label占总类label比值的开方
        Ldice = self.Ldice(pred, label, batch_size)  #
        Lcross = self.InteractLcross(realinput, reallabel, dis, Wl, label_sum)
        Lexp = self.weight[0] * Ldice + self.weight[1] * Lcross
        return Lexp
class InteractDiceLoss(nn.Module):
    def __init__(self, n_class, weights=[1, 1], gama=0.0001, alpha=1):
        super(InteractDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class
        self.gama = gama
        self.weight = weights
        self.dice = SoftDiceLoss(n_class)
        self.InteractLcross = InteractLcross(n_class, alpha)

    def forward(self, pred, label, dis):
        '''
        :param pred: batch*class*depth*length*height or batch*calss*length*height
        :param label: batch*depth*length*height or batch*length*height
        :return:
        '''
        smooth = 1
        batch_size = pred.size(0)
        one_hot_label = self.one_hot_encoder(label).contiguous().view(batch_size, self.n_class, -1)
        label_sum = torch.sum(one_hot_label, 2) + smooth  # 各类自和
        Wl = (torch.sum(label_sum) / torch.sum(label_sum, 0)) ** 0.3  # 各label占总类label比值的开方
        dice = self.dice(pred, label)
        Lcross = self.InteractLcross(pred, label, dis, Wl, label_sum)
        Lexp = self.weight[0] * dice + self.weight[1] * Lcross
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

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

def make_one_hot(pred, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         pred: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(pred.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, pred.cpu(), 1)

    return result

def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label
        input_tensor: tensor with shae [B, D, H, W]
        output_tensor: shape [B, num_class, D, H, W]
    """
    input_tensor = torch.unsqueeze(input_tensor, 1)
    tensor_list = []
    for i in range(num_class):
        temp_prob = input_tensor == i*torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim = 1)
    output_tensor = output_tensor.double()

    return output_tensor

class ExistLoss(nn.Module):
    def __init__(self,  n_class=2,smooth=0.1, axes=(-1,-2)):
        super(ExistLoss, self).__init__()
        self.smooth = smooth
        self.n_class = n_class
        self.axes = axes
    def forward(self,prediction, label):
        '''
        :param prediction: N*C*W*H*L
        :param label: N*W*H*L
        :return:
        '''
        onehot_label = get_soft_label(label,num_class=self.n_class)
        label_exist = torch.sum(onehot_label, dim=self.axes)
        prediction_exist = torch.sum(prediction, dim=self.axes)
        inter1 = label_exist/(label_exist+self.smooth)
        inter2 = prediction_exist/(prediction_exist+self.smooth)
        loss = (inter1.float()-inter2)**2
        loss = -torch.log(1-loss.mean())
        return loss
class AttentionExistExpDiceLoss(nn.Module):
    def __init__(self, n_class, alpha, gama=0.0001, weight=[1,1], dice_weight_exist=False, cross_weight_exist=True):
        super(AttentionExistExpDiceLoss, self).__init__()
        self.n_class = n_class
        self.gama = gama
        self.weight = weight
        self.alpha = alpha
        self.smooth = 1
        self.Existmap = ExistMap(n_class=n_class, alpha=1)
        self.Ldice = ExistLdice(n_class-1, self.smooth, weight_exist=dice_weight_exist)
        self.Lcross = ExistLCross(n_class-1, weight_exist=cross_weight_exist)
    def forward(self, pred, label):
        '''
        :param pred: N*C*W*H*L
        :param label: N*W*H*L
        :return:
        '''
        smooth = self.smooth
        batch_size = pred.size(0)

        onehot_label = get_soft_label(label, self.n_class)
        existmap = self.Existmap(pred, onehot_label).view(batch_size, self.n_class, -1)[:,1::] # N*C*W*H*L
        onehot_label = onehot_label.view(batch_size, self.n_class, -1)[:,1::].float()
        pred = pred.view(batch_size, self.n_class, -1)[:,1::]
        att_pred = torch.exp((pred - onehot_label) / self.alpha)*pred
        label_sum = torch.sum(onehot_label, 1) + smooth  # 非背景类label各自和
        Wl = ((torch.sum(label_sum)+smooth) / (torch.sum(label_sum, 0)+smooth))**0.5  # 各label占总非背景类label比值的开方
        Lcross = self.Lcross(pred, onehot_label, Wl, label_sum, existmap)
        Ldice = self.Ldice(att_pred, onehot_label, existmap,batch_size)

        Lexp = self.weight[0] * Ldice + self.weight[1] * Lcross
        return Lexp

class ExistMap(nn.Module):
    def __init__(self,  n_class=2,smooth=0.1, axes=(-1,-2), alpha=1.5):
        super(ExistMap, self).__init__()
        self.smooth = smooth
        self.n_class = n_class
        self.axes = axes
        self.alpha = alpha
    def forward(self,prediction, onehot_label):
        '''
        :param prediction: N*C*W*H*L
        :param label: N*C*W*H*L
        :return:
        '''
        label_exist = torch.sum(onehot_label, dim=self.axes, keepdim=True)
        prediction_exist = torch.sum(prediction, dim=self.axes, keepdim=True)
        inter1 = label_exist/(label_exist+self.smooth)
        inter2 = prediction_exist/(prediction_exist+self.smooth)
        weight = torch.abs((inter1.float()-inter2))
        weight = torch.exp(self.alpha*weight)*torch.ones_like(onehot_label).float()
        return weight
class ExistLCross(nn.Module):
    def __init__(self, n_class, weight_exist):
        super(ExistLCross, self).__init__()
        self.n_class = n_class
        self.weight_exist = weight_exist
    def forward(self, pred, label, Wl, label_sum, existmap):
        '''
        realinput: n*c*...
        reallabel: n*c*...
        Wl: 各label占总非背景类label比值的开方
        '''
        Lcross = 0
        for i in range(self.n_class):
            mask = label[:, i] == 1
            if torch.sum(mask).item() > 0:
                if self.weight_exist:
                    ExpLabel = torch.sum(-torch.log(pred[:, i][mask.detach()]+0.01)*existmap[:,i][mask.detach()])
                else:
                    ExpLabel = torch.sum(-torch.log(pred[:, i][mask.detach()] + 0.01))
                Lcross += Wl[i] * ExpLabel
        Lcross = Lcross / torch.sum(label_sum)

        return Lcross
class ExistLdice(nn.Module):
    def __init__(self, smooth, n_class, weight_exist=True):
        super(ExistLdice, self).__init__()
        self.smooth = smooth
        self.n_class = n_class
        self.weight_exist = weight_exist
    def forward(self, pred, label, existmap,batch_size):
        '''
        Ldice
        '''
        # inter = torch.sum(pred * label, 2) + self.smooth
        # union1 = torch.sum(pred, 2) + self.smooth
        # union2 = torch.sum(label, 2) + self.smooth
        if self.weight_exist:
            dice = 2.0 * (torch.sum(pred * label*existmap, 2) + self.smooth) / (torch.sum(pred*existmap, 2) + 2*self.smooth + torch.sum(label*existmap, 2))
        else:
            dice = 2.0 * (torch.sum(pred * label , 2) + self.smooth) / (
                        torch.sum(pred , 2) + 2 * self.smooth + torch.sum(label, 2))
        logdice = -torch.log(dice)
        expdice = torch.sum(logdice) # ** self.gama
        Ldice = expdice / (batch_size*self.n_class)
        return Ldice

class InteractLcross(nn.Module):
    def __init__(self, n_class, alpha):
        super(InteractLcross, self).__init__()
        self.n_class = n_class
        self.alpha = alpha
    def forward(self, realseg, reallabel, dis, Wl, label_sum):
        '''
        realinput:
        reallabel:
        Wl: 各label占总非背景类label比值的开方
        '''
        Lcross = 0
        for i in range(self.n_class):
            mask = reallabel == i
            if torch.sum(mask).item() > 0:
                ProLabel = realseg[:, i][mask.detach()]
                ProDis = dis[:, i][mask.detach()]
                LogLabel = -torch.log(ProLabel)*torch.exp(ProDis*self.alpha)
                ExpLabel = torch.sum(LogLabel)  # **self.gama
                Lcross += Wl[i] * ExpLabel
        Lcross = Lcross / torch.sum(label_sum)

        return Lcross
