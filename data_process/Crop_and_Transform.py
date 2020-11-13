#!/usr/bin/env python
import os
import numpy as np
from data_process_func import *


data_root = '/home/uestc-c1501c/LCTSC'
filename_list = ['data.nii.gz', 'label.nii.gz']
savename_list = ['crop_data_hr.nii.gz', 'crop_label_hr.nii.gz', 'crop_norm_multi_thresh_4_hr.nii.gz']
modelist = [ 'valid','train']
scale_num = [16, 16, 16]
save_as_nifty = True
respacing = True
t_spacing = [1,1,1]
norm = True
crop_by_label = True
random_pad = False
random_pad_range=[5, 20, 20]
r = [5,40,40]  #
thresh_lis = [-800, 400]
norm_lis = [0, 1]
normalize = img_multi_thresh_normalized


for mode in modelist:
    filelist =os.listdir(os.path.join(data_root, mode))
    filenum = len(filelist)
    for ii in range(filenum):
        data_path = os.path.join(data_root, mode, filelist[ii], filename_list[0])
        data_crop_save_path = os.path.join(data_root, mode, filelist[ii], savename_list[0])
        data_crop_norm_save_path = os.path.join(data_root, mode, filelist[ii], savename_list[2])

        label_path = os.path.join(data_root, mode, filelist[ii], filename_list[1])
        label_crop_save_path = os.path.join(data_root, mode, filelist[ii], savename_list[1])
        data, spacing, zoom_factor = load_nifty_volume_as_array(data_path, return_spacing=True,
                                                   respacing=respacing,target_spacing=t_spacing, order=3)
        label = np.int8(load_nifty_volume_as_array(label_path, respacing=respacing,
                                                   target_spacing=t_spacing, assigned_zoom_factor=zoom_factor,order=2,mode='label'))
        if respacing:
            spacing = t_spacing
        spacing = [spacing[2],spacing[1],spacing[0]]
        print('spacing is',spacing)
        center = data.shape[1] // 2
        rp_r = []
        if random_pad:
            for i in range(len(r)):
                rp_r.append(r[i]+random.randrange(1, random_pad_range[i], 1))
        else:
            rp_r = r
        if crop_by_label:
            [minpoint,maxpoint]=get_bound_coordinate(label, pad=rp_r)
            data_crop = data[minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]]
            if norm:
                data_crop_norm = normalize(data_crop,thresh_lis, norm_lis)
            label_crop = label[minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]]
        else:
            data_crop = data[:, center-rp_r:center+rp_r, center-rp_r:center+rp_r]
            if norm:
                data_crop_norm = normalize(data, thresh_lis, norm_lis)
            label_crop = label[:, center-rp_r:center+rp_r, center-rp_r:center+rp_r]

        print(data_crop_norm.shape)
        if save_as_nifty:
            save_array_as_nifty_volume(data_crop, data_crop_save_path,pixel_spacing=spacing)
            if norm:
                save_array_as_nifty_volume(data_crop_norm, data_crop_norm_save_path,pixel_spacing=spacing)
            save_array_as_nifty_volume(label_crop, label_crop_save_path)
            print('成功储存', filelist[ii])