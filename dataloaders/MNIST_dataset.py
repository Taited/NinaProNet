from torch.utils.data import Dataset
import struct
import numpy as np
import os
import matplotlib.pyplot as plt

filename_train_images = r'\train-images.idx3-ubyte'
filename_train_labels = r'\train-labels.idx1-ubyte'
filename_test_images = r'\t10k-images.idx3-ubyte'
filename_test_labels = r'\t10k-labels.idx1-ubyte'


class MNIST_dataset(Dataset):
    def __init__(self, root=None, train=True):
        super(MNIST_dataset)
        if not os.path.exists(root):
            raise RuntimeError('The file path did not exist.')
        self.root = root
        self.train = train
        if train:  # load train dataset
            self.data = open(root + filename_train_images, 'rb').read()
            self.targets = open(root + filename_train_labels, 'rb').read()
        else:  # load test dataset
            self.data = open(root + filename_test_images, 'rb').read()
            self.targets = open(root + filename_test_labels, 'rb').read()
        # get the length of the dataset and the size of each picture
        _, self.length, rows, columns = struct.unpack_from('>IIII', self.data, 0)
        self.size = [rows, columns]
        # since the data is bits file, the original offset need to be set
        self.original_img_offset = struct.calcsize('>IIII')
        self.original_label_offset = struct.calcsize('>II')
        self.fmt_img = '>' + str(self.size[0] * self.size[1]) + 'B'
        self.fmt_label = '>B'

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        # set img offset
        img_offset = item * struct.calcsize(self.fmt_img) + self.original_img_offset
        img = struct.unpack_from(self.fmt_img, self.data, img_offset)
        img = np.array(img).reshape((self.size[0], self.size[1]))
        # set label offset
        label_offset = item * struct.calcsize(self.fmt_label) + self.original_label_offset
        label = struct.unpack_from(self.fmt_label, self.targets, label_offset)[0]
        return img, label


if __name__ == '__main__':
    root = r'D:\Dataset\MNIST'
    myDataset = MNIST_dataset(root=root)
    img, label = myDataset[60]
    # img, label = myDataset.__getitem__(60)
    print(label)
    plt.imshow(img)
    plt.show()
