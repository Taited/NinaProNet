import os
import numpy as np
import nibabel
from data_process.data_process_func import *

ori_label = [4]
data_root = '/home/uestc-c1501c/Thoracic_OAR/'
data_mode = ['valid']


ori_labelfile_name = 'coarseg_base.nii.gz'
new_labelfile_name = 'coarseg_base_4.nii.gz'


for mode in data_mode:
    cur_data_root = os.path.join(data_root, mode)
    file_list = os.listdir(cur_data_root)

    for file in file_list:
        cur_label_path = os.path.join(cur_data_root, file, ori_labelfile_name)
        new_label_path = os.path.join(cur_data_root, file, new_labelfile_name)

        cur_label, spacing = load_nifty_volume_as_array(cur_label_path, return_spacing=True)
        new_label = np.zeros_like(cur_label)
        for i in range(len(ori_label)):
            mask = np.where(cur_label==ori_label[i])
            new_label[mask] = i+1
        save_array_as_nifty_volume(new_label, new_label_path, spacing)
        print('successfully proceed {0:}'.format(file))