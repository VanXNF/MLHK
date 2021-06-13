from math import log2
import numpy as  np


def load_dataset(dataset_dir, dataset_name, is_log=False):
    """
    加载 DNA 数据集\n
    :param dataset_dir: 数据集文件夹路径
    :param dataset_name: 数据集名称
    :param is_log: 是否输出日志
    :return: data 数据集 flag 标签
    """
    data = []
    flag = []
    with open(dataset_dir + dataset_name, 'r') as dataset:
        for data_line in dataset.readlines():
            data_line = data_line[:-2]
            data_line = data_line.split(' ')
            data.append(data_line[:-1])
            flag.append(data_line[-1])
            if is_log:
                print(f"{dataset_name}: class: {data_line[-1]}, len: {len(data_line)}, data: {data_line}")
    return data, flag


def transfer_dataset(src_dataset):
    """
    将二维数据转换为可读的 DNA 序列\n
    :param src_dataset: 原始数据集
    :return: target_dataset 转换后数据集
    """
    key_map = {'100': 'A', '010': 'C', '001': 'G', '000': 'T'}
    target_dataset = []
    for item in src_dataset:
        key = ''
        index = 0
        chips = []
        for chip in item:
            if index < 3:
                key += str(chip)
                index += 1
                if index == 3:
                    chips.append(key_map.get(key))
            else:
                key = str(chip)
                index = 1
        target_dataset.append(chips)
    return target_dataset


def cal_class(flag, is_log=False):
    """
    计算标签集中类别数量分布\n
    :param flag: 标签集
    :param is_log: 是否输出日志
    :return: flag_class 标签类别分布字典
    """
    flag_class = {}
    for it in flag:
        if flag_class.get(it) is None:
            flag_class[it] = 1
        else:
            flag_class[it] += 1
    if is_log:
        print(f"flag_class: {flag_class}")
    return flag_class


def cal_entropy(dataset, flag, is_log=False):
    """
    计算信息熵\n
    :param dataset: 数据集
    :param flag: 标签集
    :param is_log: 是否输出日志
    :return: entropy 信息熵
    """
    flag_class = cal_class(flag)
    entropy = 0
    dataset_len = len(dataset)
    for key in flag_class.keys():
        p_key = flag_class.get(key) / dataset_len
        entropy += - p_key * log2(p_key)
        if is_log:
            print(f"key {key}, num = {flag_class.get(key)}/{dataset_len},pk = {p_key}")
    return entropy


def cal_attr_num(dataset, attr_index):
    """
    计算各属性在数据集中的占比\n
    :param dataset: 数据集
    :param attr_index: 属性位置 (0-59)
    :return: [A,C,G,T] 各取值数量
    """
    attr_num = [0, 0, 0, 0]
    for item in dataset:
        if item[attr_index] == 'A':
            attr_num[0] += 1
        elif item[attr_index] == 'C':
            attr_num[1] += 1
        elif item[attr_index] == 'G':
            attr_num[2] += 1
        elif item[attr_index] == 'T':
            attr_num[3] += 1
    return attr_num


def divide_dataset(dataset, flag, attr_index):
    """
    按照对应属性划分数据集\n
    :param dataset: 数据集
    :param flag: 标签集
    :param attr_index: 属性位置 (0-59)
    :return: datasets 划分后数据集字典，划分后的标签集字典
    """
    attr_values = ['A', 'C', 'G', 'T']
    datasets = {}
    flags = {}
    for attr in attr_values:
        datasets[attr] = []
        flags[attr] = []
    for index in range(0, len(dataset)):
        data = dataset[index]
        for attr in attr_values:
            if data[attr_index] == attr:
                datasets.get(attr).append(data)
                flags.get(attr).append(flag[index])
    return datasets, flags


def cal_gain(dataset, flag, attr_index):
    dataset_len = len(dataset)
    entropy = cal_entropy(dataset, flag, True)
    attr_values = ['A', 'C', 'G', 'T']
    datasets, flags = divide_dataset(dataset, flag, attr_index)
    gain = entropy
    for attr in attr_values:
        dataset_attr = datasets.get(attr)
        flag_attr = flags.get(attr)
        gain += - (len(dataset_attr) / dataset_len) * cal_entropy(dataset_attr, flag_attr)
    return gain


def tree_generate(dataset, flag, key, father_attr_index=-1):
    node_list = []
    node = None
    flag_class = cal_class(flag)
    if len(flag_class.keys()) == 1:
        print("only 1 class")
        node = [key, -1, father_attr_index, list(flag_class.keys())[0]]


if __name__ == '__main__':
    # 读取数据集
    dataset_train, flag_train = load_dataset(dataset_dir='./datasets/',
                                             dataset_name='dna.data',
                                             is_log=False)
    dataset_test, flag_test = load_dataset(dataset_dir='./datasets/',
                                           dataset_name='dna.test',
                                           is_log=False)
    # 转换处理数据集
    dataset_train = transfer_dataset(dataset_train)
    dataset_test = transfer_dataset(dataset_test)

    # e = cal_entropy(dataset_train, flag_train)
    print(cal_attr_num(dataset_train, 1))
    # divide_dataset(dataset_train, 1)
    max_gain = 0
    max_index = 0
    for i in range(0, 60):
        gain = cal_gain(dataset_train, flag_train, i)
        if max_gain < gain:
            max_gain = gain
            max_index = i
    print(max_gain, max_index)

    # a = {'A': 77}
    # print(list(a.keys())[0])
