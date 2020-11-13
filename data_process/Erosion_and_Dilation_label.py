import os
from skimage import morphology
from scipy import ndimage
import numpy as np
from data_process_func import *
data_root = '/home/uestc-c1501c/LCTSC'
filename_list = ['crop_label_mr.nii.gz', 'crop_label_hr.nii.gz']
savename_list = ['crop_label_mr.nii.gz', 'crop_label_hr.nii.gz']
modelist = [ 'valid','train']

for mode in modelist:
    filelist =os.listdir(os.path.join(data_root, mode))
    filenum = len(filelist)
    for ii in range(filenum):
        for iii in range(len(filename_list)):
            label_path = os.path.join(data_root, mode, filelist[ii], filename_list[iii])
            label_save_path = os.path.join(data_root, mode, filelist[ii], savename_list[iii])
            label,spacing = load_nifty_volume_as_array(label_path,return_spacing=True)
            spacing.reverse()
            erosion_label = Erosion_Multi_label(label, np.ones([1,3,3]), class_number=np.max(label)+1)
            dilation_label = np.int8(Dilation_Multi_label(erosion_label, np.ones([1, 4, 4]), class_number=np.max(label) + 1))
            save_array_as_nifty_volume(dilation_label, label_save_path, pixel_spacing=spacing)
            print('Erosion and dilation:',label_path)