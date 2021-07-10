# -*- coding: utf-8 -*-
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


class FeatherImg(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(FeatherImg, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            file_annotation = root + '/train.json'
            img_folder = root + '/train_set/'
        else:
            file_annotation = root + '/test.json'
            img_folder = root + '/test_set/'

        fp = open(file_annotation, 'r')
        data_dict = json.load(fp)

        # 如果图像数和标签数不匹配说明数据集标注生成有问题，报错提示
        # train 4674 test 1999
        assert len(data_dict['image']) == len(data_dict['categoary'])
        num_data = len(data_dict['image'])

        self.filenames = []
        self.labels = []
        self.img_folder = img_folder
        for i in range(num_data):
            #             print(data_dict['image'].get(str(i)))
            data_id = int(data_dict['categoary'].get(str(i)))
            #             if data_id == 56:
            #                 data_id = 5
            self.filenames.append(data_dict['image'].get(str(i)))
            self.labels.append(data_id - 1)

    def __getitem__(self, index):
        #         print(index)
        img_name = f'{self.img_folder}{self.filenames[index]}'
        label = self.labels[index]
        #         print(label)

        img = plt.imread(img_name)
        img = self.transform(np.array(img))  # 可以根据指定的转化形式对数据集进行转换

        # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
        return img, torch.tensor(label)

    def __len__(self):
        return len(self.filenames)
