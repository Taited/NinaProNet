import os
from skimage import morphology
from scipy import ndimage
import numpy as np
from data_process_func import *
data_root = '/home/uestc-c1501c/LCTSC'
low_filename = 'crop_label_lr.nii.gz'
filename_dic = {'crop_label_mr.nii.gz':2, 'crop_label_hr.nii.gz':4,
                'crop_norm_multi_thresh_4_mr.nii.gz':2,'crop_norm_multi_thresh_4_hr.nii.gz':4}
modelist = [ 'valid','train']

for mode in modelist:
    filelist =os.listdir(os.path.join(data_root, mode))
    filenum = len(filelist)
    for ii in range(filenum):
        low_label_path = os.path.join(data_root, mode, filelist[ii], low_filename)
        label,spacing = load_nifty_volume_as_array(low_label_path, return_spacing=True)
        low_w = label.shape[0]
        for filename in filename_dic.keys():
            data_path = os.path.join(data_root, mode, filelist[ii], filename)
            data_save_path = os.path.join(data_root, mode, filelist[ii], filename)
            data,spacing = load_nifty_volume_as_array(data_path,return_spacing=True)
            cur_w = data.shape[0]
            if cur_w != filename_dic[filename]*low_w:
                spacing.reverse()
                zoom_factor = [filename_dic[filename]*low_w/cur_w, 1, 1]
                if 'label' in filename:
                    data = resize_Multi_label_to_given_shape(data,
                                                             zoom_factor=zoom_factor, class_number=np.max(data) + 1,
                                                             order=1)
                else:
                    data = ndimage.zoom(data, zoom=zoom_factor, order=1)
                print(low_w, data.shape[0])
                save_array_as_nifty_volume(data, data_save_path, pixel_spacing=spacing)
                print('Zoomed:',data_path)