#!/usr/bin/env python
import os
import numpy as np
from data_process.data_process_func import *

class Adapt_transform(object):
    def __init__(self, hu_lis=[0], norm_lis=[0], smooth_lis=[0], norm=False):
        super(Adapt_transform, self).__init__()
        self.hu_lis = np.asarray(hu_lis)
        self.norm_lis = np.asarray(norm_lis)
        self.smooth_lis = np.asarray(smooth_lis)
        self.norm = norm
    def forward(self, img):
        cur_file = np.zeros_like(img).astype(np.float)
        for i in range(self.hu_lis.shape[0]):  # 对一类中所有hu遍历
            cur_file += np.abs(self.norm_lis[i])/(1+np.exp((-img+100*np.sum(self.hu_lis[0:i+1]))/(100*np.abs(self.smooth_lis[i])+30)))
        if self.norm:
            cur_file = (cur_file-np.min(cur_file))/(np.max(cur_file)-np.min(cur_file))
        new_file = cur_file
        return new_file

data_root = '/home/uestc-c1501c/Thoracic_OAR/'
filename_list = ['data.nii.gz', 'label.nii.gz']
savename_list = ['crop_data.nii.gz', 'crop_label.nii.gz', 'crop_norm_norm_4t.nii.gz']
modelist = [ 'valid']
scale_num = [16, 16, 16]
save_as_nifty = True
respacing = False
norm = True
crop_by_label = True
r = [5,40,40]
hu_lis = [-0.16736238, 1.0182023,  0.70408607, -0.25095922]
norm_lis = [ 3.0769236, -2.3885267, -1.0168839,  0.08459189]
smooth_lis = [-1.0529016,  4.542475,  -0.5822441,  1.6142025]
adapt_trans = Adapt_transform(hu_lis=hu_lis, norm_lis=norm_lis, smooth_lis=smooth_lis)
for mode in modelist:
    filelist =os.listdir(os.path.join(data_root, mode))
    filenum = len(filelist)
    for ii in range(filenum):
        data_path = os.path.join(data_root, mode, filelist[ii], filename_list[0])
        data_crop_save_path = os.path.join(data_root, mode, filelist[ii], savename_list[0])
        data_crop_norm_save_path = os.path.join(data_root, mode, filelist[ii], savename_list[2])

        label_path = os.path.join(data_root, mode, filelist[ii], filename_list[1])
        label_crop_save_path = os.path.join(data_root, mode, filelist[ii], savename_list[1])
        data, spacing = load_nifty_volume_as_array(data_path, return_spacing=True)
        spacing.reverse()
        label = np.int8(load_nifty_volume_as_array(label_path))
        center = data.shape[1] // 2
        if crop_by_label:
            [minpoint,maxpoint]=get_bound_coordinate(label, pad=r)
            data_crop = data[minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]]
            if norm:
                data_crop_norm = adapt_trans.forward(data_crop)
            label_crop = label[minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]]
        else:
            data_crop = data[:, center-r:center+r, center-r:center+r]
            if norm:
                data_crop_norm = adapt_trans.forward(data)
            label_crop = label[:, center-r:center+r, center-r:center+r]
        if save_as_nifty:
            if norm:
                save_array_as_nifty_volume(data_crop_norm, data_crop_norm_save_path,pixel_spacing=spacing)
        print('成功储存', filelist[ii])