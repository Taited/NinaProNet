import os
import numpy as np
import nibabel
from data_process_func import *

ori_label = [20]
data_root = '/lyc/MICCAI-19-StructSeg/HaN_OAR_center_crop'
data_mode = ['train', 'valid']

ori_datafile_name = 'crop_data_-500-800.nii.gz'
new_datafile_name = 'crop_spinal_data_-500-800.nii.gz'

ori_probfile_name = 'Prob.nii.gz'
new_probfile_name = 'crop_spinal_prob_-500-800.nii.gz'

ori_labelfile_name = 'crop_label.nii.gz'
new_labelfile_name = 'crop_spinal_label.nii.gz'

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
        cur_prob_path = os.path.join(cur_data_root, file, ori_probfile_name)
        new_prob_path = os.path.join(cur_data_root, file, new_probfile_name)

        data = load_nifty_volume_as_array(cur_data_path)
        prob = load_nifty_volume_as_array(cur_prob_path, transpose=False)
        cur_label = load_nifty_volume_as_array(cur_label_path)

        new_label = np.zeros_like(cur_label)
        for i in range(len(ori_label)):
            mask = cur_label == ori_label[i]
            [minpoint, maxpoint] = get_bound_coordinate(mask, pad=[8, 8, 16])
            for ii in range(len(minpoint)):
                min_cor[ii] = min(min_cor[ii], minpoint[ii])
                max_cor[ii] = max(max_cor[ii], maxpoint[ii])
            new_label[mask] = i + 1


        length = [max_cor[j] - min_cor[j] for j in range(len(max_cor))]
        max_cor = [max_cor[iii] + scale_num[iii] - length[iii] % scale_num[iii] for iii in range(len(length))]
        for i in range(len(length)):
            if max_cor[i] > data.shape[i]:
                min_cor[i] = min_cor[i] - max_cor[i] + data.shape[i]
                max_cor[i] = data.shape[i]

        print(max_cor, min_cor)


        new_data = data[min_cor[0]:max_cor[0], min_cor[1]:max_cor[1], min_cor[2]:max_cor[2]].astype(np.float32)
        new_label = new_label[min_cor[0]:max_cor[0], min_cor[1]:max_cor[1], min_cor[2]:max_cor[2]]
        new_label = np.int8(new_label)
        new_prob = prob[ori_label, min_cor[0]:max_cor[0], min_cor[1]:max_cor[1], min_cor[2]:max_cor[2]]
        new_prob = np.vstack((new_data[np.newaxis,:], new_prob))

        save_array_as_nifty_volume(new_label, new_label_path)
        save_array_as_nifty_volume(new_data, new_data_path)
        save_array_as_nifty_volume(new_prob, new_prob_path, transpose=False, pixel_spacing=[1,3,1,1])
        print('successfully proceed {0:}'.format(file))