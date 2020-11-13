# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 23:17:09 2019
An attempt to reimplement
Shape-Aware Organ Segmentation by Predicting Signed Distance Maps (AAAI 2020)
https://arxiv.org/abs/1912.03849

## usage
criterion_sdf = SDFLoss().cuda()
criterion_dice = DiceLoss().cuda()
# create model
net = UNet(n_channels=3, n_classes=num_classes)

outputs_sdf = net(inputs)
# torch.Size([b, 2, x, y]) torch.Size([b, 1, x, y])
# print('outputs_sdf.shape, label.shape: ', outputs_sdf.shape, label.shape)
loss_sdf = criterion_sdf(outputs_sdf, label)
outputs = 1.0 / (1.0 + torch.exp(-1500.0 * outputs_sdf))
loss_dice = criterion_dice(outputs[:,1:2,...], label)
loss = loss_dice + 10.0 * loss_sdf


@author: Jun Ma
"""


import numpy as np
from scipy.ndimage import distance_transform_edt as distance
import torch
from torch import nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1e-5
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / \
            (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return dice

def compute_sdm(segmentation):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the Signed Distance Map (SDM)
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation

    """
    # print(type(segmentation), segmentation.shape)
    segmentation = segmentation.astype(np.uint8)
    normalized_sdm = np.zeros(segmentation.shape)
    for b in range(segmentation.shape[0]): # batch size
        for c in range(0, segmentation.shape[1]): # class_num
            # ignore background
            posmask = segmentation[b][c]
            negmask = ~posmask
            sdm = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
            normalized_sdm[b][c] = 2.0/(np.max(sdm) - np.min(sdm))*(sdm-np.min(sdm)) - 1.0
    return normalized_sdm

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


class SDFLoss(nn.Module):
    def __init__(self):
        super(SDFLoss, self).__init__()

    def forward(self, net_output, gt):
        """
        net_output: net logits; shape=(batch_size, class, x, y, z)
        gt: ground truth; (shape (batch_size, 1, x, y, z) OR (batch_size, x, y, z))
        """
        smooth = 1e-5
        axes = tuple(range(2, len(net_output.size())))
        shp_x = net_output.shape
        shp_y = gt.shape

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)
            gt_sdm_npy = compute_sdm(y_onehot.cpu().numpy())
            gt_sdm = torch.from_numpy(gt_sdm_npy).float().cuda(net_output.device.index)
        intersect = sum_tensor(net_output * gt_sdm, axes, keepdim=False)
        pd_sum = sum_tensor(net_output ** 2, axes, keepdim=False)
        gt_sum = sum_tensor(gt_sdm ** 2, axes, keepdim=False)
        L_product = (intersect + smooth) / (intersect + pd_sum + gt_sum)
        # print('L_product.shape', L_product.shape) (batch_size, class_num)
        L_SDM = - L_product.mean() + torch.norm(net_output - gt_sdm, 1)

        return L_SDM