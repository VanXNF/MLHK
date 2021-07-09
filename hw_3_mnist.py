# -*- coding: utf-8 -*-
"""
========================
关于IDX文件格式的解析规则：
========================
THE IDX FILE FORMAT
the IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.
The basic format is
magic number
size in dimension 0
size in dimension 1
size in dimension 2
.....
size in dimension N
data
The magic number is an integer (MSB first). The first 2 bytes are always 0.
The third byte codes the type of the data:
0x08: unsigned byte
0x09: signed byte
0x0B: short (2 bytes)
0x0C: int (4 bytes)
0x0D: float (4 bytes)
0x0E: double (8 bytes)
The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....
The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).
The data is stored like in a C array, i.e. the index in the last dimension changes the fastest.
"""

import numpy as np
import struct
import matplotlib.pyplot as plt

data_path = 'datasets/MNIST'
# 训练集文件
train_images_idx3_ubyte_file = f'{data_path}/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = f'{data_path}/train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = f'{data_path}/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = f'{data_path}/t10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file, is_log=False):
    """
    解析 idx3文件的通用函数\n
    :param idx3_ubyte_file: idx3文件路径
    :param is_log: 是否输出日志
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    if is_log:
        print(f'magic number:{magic_number}, image number: {num_images}, image size: {num_rows}*{num_cols}')

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if is_log:
            if (i + 1) % 10000 == 0:
                print(f'already decoded {i + 1}')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file, is_log=False):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :param is_log 是否输出日志
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    if is_log:
        print(f'magic number: {magic_number}, image number: {num_images}')

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if is_log:
            if (i + 1) % 10000 == 0:
                print(f'already decoded {i + 1}')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file, is_log=False):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    :param idx_ubyte_file: idx文件路径
    :param is_log: 是否输出日志
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file, is_log)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file, is_log=False):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.
    :param idx_ubyte_file: idx文件路径
    :param is_log: 是否输出日志
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file, is_log)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file, is_log=False):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    :param idx_ubyte_file: idx文件路径
    :param is_log: 是否输出日志
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file, is_log)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file, is_log=False):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.
    :param idx_ubyte_file: idx文件路径
    :param is_log: 是否输出日志
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file, is_log)


if __name__ == '__main__':
    train_images = load_train_images()
    print(type(train_images), train_images.shape)
    train_labels = load_train_labels()
    print(type(train_labels), train_labels.shape)
    test_images = load_test_images()
    print(type(test_images), test_images.shape)
    test_labels = load_test_labels()
    print(type(test_labels), test_labels.shape)

    # 查看前十个数据及其标签以读取是否正确
    for it in range(10):
        print(train_labels[it])
        print(np.max(train_images), np.min(train_images))
        plt.imshow(train_images[it], cmap='gray')
        plt.show()
    print('done')
