import GeodisTK
from scipy import ndimage
import os
from data_process.data_process_func import load_nifty_volume_as_array, save_array_as_nifty_volume
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import sys


def geodesic_distance(img, seed, distance_threshold):
    threshold = distance_threshold
    if (seed.sum() > 0):
        geo_dis = GeodisTK.geodesic3d_raster_scan(img, seed, 1.0, 2)
        geo_dis[np.where(geo_dis > threshold)] = threshold
        dis = 1 - geo_dis / threshold  # recale to 0-1
    else:
        dis = np.zeros_like(img, np.float32)
    return dis


def get_center_cor(img):
    '''
    get 2d binary img center corardiate
    :param img: 2d binary img
    :return:
    '''
    contours, cnt = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])  # 计算第一条轮廓的各阶矩，字典模式
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    return center_x, center_y


def get_sin(img):
    contours, cnt = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours).squeeze()  # contours is a n*2 array
    # init as the index of x and y. accelerate calculation
    point_index = np.array([np.zeros_like(img), np.zeros_like(img)]).astype(float)
    for x in range(point_index.shape[1]):
        point_index[0, x, :] = x
    for y in range(point_index.shape[2]):
        point_index[1, :, y] = y
    if contours is not None:
        point_dis = np.zeros([contours.shape[0], img.shape[0], img.shape[1]])
        point_sin = np.zeros([contours.shape[0], img.shape[0], img.shape[1]])
        for contours_id, (contours_x, contours_y) in enumerate(contours):
            x = contours_x - point_index[0]
            y = contours_y - point_index[1]
            point_dis[contours_id] = np.sqrt(np.square(x) + np.square(y))
            ems = 1e-4
            point_sin[contours_id] = (y + ems) / (point_dis[contours_id] + ems)

    min_id = np.argmin(point_dis, axis=0)
    min_sin = np.zeros_like(img).astype(float)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            min_sin[x, y] = point_sin[min_id[x, y], x, y]
    margin = 128 + 80
    plt.imshow(min_sin[margin+10:-margin, margin+30:-margin])
    plt.show()
    return min_sin


def get_contours(img, is_show=False):
    img = img.astype(np.uint8)  # 必须转换成uint8，否则cv2会报错
    kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (3, 3))
    eroded_img = cv2.erode(img, kernel)
    contour = img - eroded_img
    if is_show:
        margin = 200
        _, plots = plt.subplots(1, 3)
        plots[0].imshow(img[margin:-margin, margin:-margin])
        plots[1].imshow(eroded_img[margin:-margin, margin:-margin])
        plots[2].imshow(contour[margin:-margin, margin:-margin])
        plt.show()
    return contour


def get_vector3d_fake(img):
    MAX_DISTANCE = 50
    vector_x = MAX_DISTANCE * np.ones_like(img).astype(float)
    vector_y = MAX_DISTANCE * np.ones_like(img).astype(float)
    vector_z = MAX_DISTANCE * np.ones_like(img).astype(float)
    # 直接建立起所有边界点的索引
    contours = np.nonzero(img)
    total_distance = np.zeros([contours[0].size, img.shape[0], img.shape[1], img.shape[2]], dtype=float)
    start = time.process_time()
    for i in range(contours[0].size):
        total_distance[i, :, :, :] = get_distance_3d([contours[0][i], contours[1][i], contours[2][i]], img)
        print('{}%'.format(i*100/contours[0].size))
    end = time.process_time()
    print('time of distance map: %s Seconds' % (end - start))
    the_min = np.min(total_distance, axis=0).astype(int)
    total_distance = 1
    for id_z in range(img.shape[0]):
        print('{}%'.format(id_z*100/img.shape[0]))
        for id_x in range(img.shape[1]):
            for id_y in range(img.shape[2]):
                contour_id = the_min[id_z, id_x, id_y]
                vector_z[id_z, id_x, id_y] = contours[0][contour_id] - id_z
                vector_x[id_z, id_x, id_y] = contours[1][contour_id] - id_x
                vector_y[id_z, id_x, id_y] = contours[2][contour_id] - id_y
    return np.array([vector_z, vector_x, vector_y])


def get_vector3d(img):
    MAX_DISTANCE = 50
    vector_x = MAX_DISTANCE * np.ones_like(img).astype(float)
    vector_y = MAX_DISTANCE * np.ones_like(img).astype(float)
    vector_z = MAX_DISTANCE * np.ones_like(img).astype(float)
    # 直接建立起所有边界点的索引
    contours = np.nonzero(img)
    distance_id = np.zeros_like(img).astype(int)
    start = time.process_time()
    old_distance = get_distance_3d([contours[0][0], contours[1][0], contours[2][0]], img)
    for i in range(contours[0].size):
        new_distance = get_distance_3d([contours[0][i], contours[1][i], contours[2][i]], img)
        the_id = np.where(new_distance < old_distance)
        distance_id[the_id] = i
        old_distance[the_id] = new_distance[the_id]
        sys.stdout.write('\r' + 'distance map: {:.2f}%'.format(i*100/contours[0].size))
        sys.stdout.flush()
    end = time.process_time()
    print('time of distance map: %s Seconds' % (end - start))

    for id_z in range(img.shape[0]):
        sys.stdout.write('\r' + 'vector process: {:.2f}%'.format(id_z*100/img.shape[0]))
        sys.stdout.flush()
        for id_x in range(img.shape[1]):
            for id_y in range(img.shape[2]):
                contour_id = distance_id[id_z, id_x, id_y]
                vector_z[id_z, id_x, id_y] = contours[0][contour_id] - id_z
                vector_x[id_z, id_x, id_y] = contours[1][contour_id] - id_x
                vector_y[id_z, id_x, id_y] = contours[2][contour_id] - id_y
    return np.array([vector_z, vector_x, vector_y])


# 计算的是边界点到全局所有点的距离
def get_distance_3d(seed, img, spacing=[1, 1, 1]):
    # 提前创建索引，以内存换速度
    x = np.zeros([img.shape[1], img.shape[2]]).astype(float)
    y = np.zeros([img.shape[1], img.shape[2]]).astype(float)
    for _ in range(img.shape[1]):
        x[_, :] = _
    for _ in range(img.shape[2]):
        y[:, _] = _

    distance = np.zeros_like(img).astype(float)
    for id_z in range(img.shape[0]):
        temp_z = spacing[0] * (seed[0] - id_z)
        temp_x = spacing[1] * (seed[1] - x)
        temp_y = spacing[2] * (seed[2] - y)
        distance[id_z, :, :] = np.sqrt(np.square(temp_z) + np.square(temp_x) + np.square(temp_y))
    return distance


def get_vector2d(img):
    MAX_DISTANCE = 50
    index_x = np.zeros([img.shape[1], img.shape[2]]).astype(float)
    index_y = np.zeros([img.shape[1], img.shape[2]]).astype(float)
    for x in range(img.shape[1]):
        index_x[x, :] = x
    for y in range(img.shape[2]):
        index_y[:, y] = y

    vector = np.zeros([2, img.shape[0], img.shape[1], img.shape[2]]).astype(float)

    for id_z in range(img.shape[0]):
        if np.sum(img[id_z]) == 0:
            vector[:, id_z, :, :] = 0  # MAX_DISTANCE
        else:
            contours = np.nonzero(img[id_z])
            point_dis = np.zeros([contours[0].shape[0], img.shape[1], img.shape[2]])
            for contours_id in range(contours[0].shape[0]):
                contours_x, contours_y = contours[0][contours_id], contours[1][contours_id]
                x = contours_x - index_x
                y = contours_y - index_y
                point_dis[contours_id] = np.sqrt(np.square(x) + np.square(y))
            min_id = np.argmin(point_dis, axis=0)
            for x in range(img.shape[1]):
                for y in range(img.shape[2]):
                    vector[0, id_z, x, y] = contours[0][min_id[x, y]] - index_x[x, y]
                    vector[1, id_z, x, y] = contours[1][min_id[x, y]] - index_y[x, y]
    vector[np.where(vector > MAX_DISTANCE)] = MAX_DISTANCE
    return vector


'''
读取label，根据label在每张slice中心，生成距离图
'''
if __name__ == '__main__':
    # data_root = '/home/gd/Data/test_delete_after'
    data_root = r'D:\Dataset\test_delete_after'
    mode = ['train', 'valid']
    show = False  # 是否可视化康康
    geo = False
    blur = True
    degree_savename = 'szwt_vector3d_{}.nii.gz'  # 存储距离图的名称
    for submode in mode:
        for item in os.listdir(os.path.join(data_root, submode)):
            item_path = os.path.join(data_root, submode, item)
            label = load_nifty_volume_as_array(os.path.join(item_path, 'crop_label.nii.gz'))
            mask = np.where(label == 1)
            nlabel = np.zeros_like(label)
            center = np.zeros([label.shape[0], 2])
            nlabel[mask] = 1
            print(item, np.sum(nlabel))

            start = time.process_time()
            image = load_nifty_volume_as_array(os.path.join(item_path, 'crop_data.nii.gz'))
            contours = np.zeros_like(nlabel).astype(float)
            for i in range(nlabel.shape[0]):
                contours[i] = get_contours(nlabel[i])
            vector = get_vector3d(contours)
            end = time.process_time()
            print('Total Running time: {} Seconds'.format(end-start))

            margin = 128 + 80
            if show:
                f, plots = plt.subplots(1, 3)
                plots[0].imshow(label[-36, margin:-margin, margin:-margin])
                plots[1].imshow(vector[0, -36, margin:-margin, margin:-margin])
                plots[2].imshow(vector[1, -36, margin:-margin, margin:-margin])
                plt.show()
            save_array_as_nifty_volume(vector[0], os.path.join(item_path, degree_savename.format('z')))
            save_array_as_nifty_volume(vector[1], os.path.join(item_path, degree_savename.format('x')))
            save_array_as_nifty_volume(vector[2], os.path.join(item_path, degree_savename.format('y')))
