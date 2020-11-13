import os
import numpy as np
import nibabel
from data_process.data_process_func import *


ori_label = [1]
data_root = '/lyc/MICCAI-19-StructSeg/HaN_OAR_center_crop'
data_mode = ['valid']

ori_datafile_name = 'data.nii.gz'
new_datafile_name = 'crop_gtv_data.nii.gz'

ori_labelfile_name = 'gtv_label.nii.gz'
new_labelfile_name = 'crop_gtv_label.nii.gz'
thresh_lis = [-500, -100, 100, 600, 1500]
norm_lis = [0, 0.1, 0.7, 0.9, 1]

scale_num = [16, 16, 16]

for mode in data_mode:
        cur_data_root = os.path.join(data_root, mode)
        file_list = os.listdir(cur_data_root)


        for file in file_list:
            min_cor = [512, 512, 512]
            max_cor = [0, 0, 0]

            cur_data_path = os.path.join(cur_data_root, file, ori_datafile_name)
            new_data_path = os.path.join(cur_data_root, file, new_datafile_name)
            cur_label_path = os.path.join(cur_data_root, file, ori_labelfile_name)
            new_label_path = os.path.join(cur_data_root, file, new_labelfile_name)

            data = load_nifty_volume_as_array(cur_data_path)
            data = img_multi_thresh_normalized(data,thresh_lis=thresh_lis, norm_lis=norm_lis)
            cur_label = load_nifty_volume_as_array(cur_label_path)

            new_label = np.zeros_like(cur_label)
            for i in range(len(ori_label)):
                mask = cur_label==ori_label[i]
                [minpoint, maxpoint] = get_bound_coordinate(mask, pad=[8, 32, 32])
                for ii in range(len(minpoint)):
                    min_cor[ii]=min(min_cor[ii], minpoint[ii])
                    max_cor[ii]=max(max_cor[ii], maxpoint[ii])
                new_label[mask]=i+1
                
            length = [max_cor[j]-min_cor[j] for j in range(len(max_cor))]
            max_cor = [max_cor[iii]+scale_num[iii]-length[iii]%scale_num[iii] for iii in range(len(length))]
            
            new_data = data[min_cor[0]:max_cor[0], min_cor[1]:max_cor[1], min_cor[2]:max_cor[2]]
            new_label = new_label[min_cor[0]:max_cor[0], min_cor[1]:max_cor[1], min_cor[2]:max_cor[2]]
            new_label = np.int8(new_label)
            
            save_array_as_nifty_volume(new_label, new_label_path)
            save_array_as_nifty_volume(new_data, new_data_path)
            print('successfully proceed {0:}'.format(file))