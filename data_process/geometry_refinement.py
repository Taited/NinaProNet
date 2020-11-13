# -*- coding: utf-8 -*-
#!/usr/bin/env python
from __future__ import absolute_import, print_function
import time
import skimage
import numpy as np
from data_process.data_process_func import get_largest_two_component


def continuty_refinement(prediction, probability, slice_count_thresh=10, prob_thresh=0.001):
    '''
    在分割的最大最小slice间，对间断的slice增大概率图，以起到联通作用。
    :param prediction: h*w*l
    :param probability: h*w*l
    :return:
    '''
    prediction = get_largest_two_component(prediction, threshold=50) ##去除小区域噪音
    slice_count = np.sum(prediction, axis=(-1,-2))
    slice_continuty = np.where(slice_count>slice_count_thresh)[0]
    if len(slice_continuty)>2:
        slice_break = np.where(slice_count<=slice_count_thresh)[0]
        slice_min = np.min(slice_continuty)
        slice_max = np.max(slice_continuty) #正常分割上下界
        mask = np.where((slice_break<slice_max)&(slice_break>slice_min))[0]  #正常分割内的间断slice
        if len(mask)>0:
            slice_break = slice_break[mask]
            print('max', slice_max, 'min', slice_min, 'mask', mask)
            for slice in slice_break: ###每层断片
                probmask = np.where(probability[slice]>=prob_thresh)
                if len(probmask[0])>0:
                    probsum = np.sum(probability[slice][probmask])
                    k = probmask[0].shape[0]/probsum
                    probability[slice][probmask]*=k
            print('refinement accomplish！')

    return probability

def distance_refinement(prediction, probability,  slice_count_thresh=10, prob_base=100):
    '''
    在分割的最大最小slice间，对间断的slice增大概率图，以起到联通作用。
    :param prediction: h*w*l
    :param probability: h*w*l
    :return:
    '''
    prediction = get_largest_two_component(prediction, threshold=50) ##去除小区域噪音
    slice_count = np.sum(prediction, axis=(-1,-2))
    slice_continuty = np.where(slice_count>slice_count_thresh)[0]
    if len(slice_continuty)>2:
        slice_break = np.where(slice_count<=slice_count_thresh)[0]
        slice_min = np.min(slice_continuty)
        slice_max = np.max(slice_continuty) #正常分割上下界
        mask = np.where((slice_break<slice_max)&(slice_break>slice_min))[0]  #正常分割内的间断slice
        if len(mask)>0:
            slice_break = slice_break[mask]
            print('max', slice_max, 'min', slice_min, 'mask', mask)
            for slice in slice_break: ###每层断片
                prob_thresh = np.max(probability[slice])/prob_base
                probmask = np.where(probability[slice]>=prob_thresh)
                if len(probmask[0])>0:
                    probsum = np.sum(probability[slice][probmask])
                    k = probmask[0].shape[0]/probsum
                    probability[slice][probmask]*=k
                    # probability[slice]+=distance[slice]
            print('refinement accomplish！')

    return probability



