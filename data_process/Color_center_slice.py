#!/usr/bin/env python
import os
import numpy as np
from data_process.data_process_func import *


data_root = '/home/uestc-c1501c/StructSeg/Lung_GTV_n/'
filename_list = ['crop_label.nii.gz']
savename_list = ['crop_center_label.nii.gz']
modelist = [ 'valid','train']
save_as_nifty = True
norm_lis = [0, 1]
normalize = img_multi_thresh_normalized

for mode in modelist:
    filelist =os.listdir(os.path.join(data_root, mode))
    filenum = len(filelist)
    for ii in range(filenum):
        label_path = os.path.join(data_root, mode, filelist[ii], filename_list[0])
        center_label_save_path = os.path.join(data_root, mode, filelist[ii], savename_list[0])
        label = np.int8(load_nifty_volume_as_array(label_path))
        center_label = np.zeros_like(label)
        slice_sum = np.sum(label, axis=(1,2))
        max_slice = np.argmax(slice_sum)
        center_label[max_slice]=label[max_slice]
        if save_as_nifty:
            save_array_as_nifty_volume(center_label, center_label_save_path)
        print('成功储存', filelist[ii])