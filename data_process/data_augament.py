import  os
import numpy as np
from data_process.data_process_func import *
from scipy import ndimage

def zoom_data(data_root,save_root, zoom_factor, class_number, filename):
    """
    对数据进行插值并储存，
    :param data_root: 数据所在上层目录
    :param save_root: 存储的顶层目录
    :zoom_factor:   缩放倍数
    :return:
    """
    # for mode in os.listdir(data_root):
    mode = 'test'
    for data_name in os.listdir(data_root+'/'+mode):
        data_path = data_root+'/'+mode+'/'+data_name
        zoom_path = save_root+'/'+'train'+'/'+data_name
        mkdir(zoom_path)
        original_img = np.load(data_path+'/'+filename[0])
        original_label = np.int16(np.load(data_path+'/'+filename[1]))
        zoom_label = np.int16(resize_Multi_label_to_given_shape(original_label, zoom_factor, class_number, order=3))
        zoom_img = ndimage.interpolation.zoom(original_img, zoom_factor, order = 3)
        np.save(zoom_path+'/'+filename[0], zoom_img)
        np.save(zoom_path+'/'+filename[1], zoom_label)
        print('已完成对%s的插值' % data_name)
    return

if __name__ =='__main__':
    data_root = '/lyc/RTData/3D_data'
    save_root = '/lyc/RTData/3D_zoom'
    zoom_factor = [4, 1, 1]
    class_number = 5
    file_name = ['Img.npy', 'label.npy']
    zoom_data(
        data_root = '/lyc/RTData/3D_data',
        save_root = '/lyc/RTData/3D_zoom',
        zoom_factor = [4, 1, 1],
        class_number = 5,
        filename = ['Img.npy', 'label.npy'])

