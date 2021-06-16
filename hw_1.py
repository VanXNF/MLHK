# -*- coding: utf-8 -*-
from math import log2
import numpy as np
from graphviz import Digraph


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


def cal_max_class(flag, flag_class=None, is_log=False):
    """
    计算标签集中类别数量分布\n
    :param flag: 标签集
    :param flag_class: 标签类别分布
    :param is_log: 是否输出日志
    :return: 最多分布的类
    """
    if flag_class is None:
        flag_class = cal_class(flag, is_log)
    max_key = ''
    max_class_num = 0
    for key in flag_class.keys():
        if max_class_num < flag_class.get(key):
            max_key = key
            max_class_num = flag_class.get(key)
    if is_log:
        print(f"max class: {max_key}, max class num: {max_class_num}")
    return max_key


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
    entropy = cal_entropy(dataset, flag)
    attr_values = ['A', 'C', 'G', 'T']
    datasets, flags = divide_dataset(dataset, flag, attr_index)
    gain = entropy
    for attr in attr_values:
        dataset_attr = datasets.get(attr)
        flag_attr = flags.get(attr)
        gain += - (len(dataset_attr) / dataset_len) * cal_entropy(dataset_attr, flag_attr)
    return gain


def create_node(key, attr_index, father_attr_index, node_class, node_number, elements):
    """
    创建节点\n
    :param key: 当前节点关键字
    :param attr_index: 当前节点划分属性位置
    :param father_attr_index: 父节点划分属性位置
    :param node_class: 节点类别
    :param node_number: 节点内元素数量
    :param elements: 元素列表
    :return:
    """
    return [key, attr_index, father_attr_index, node_class, node_number, elements]


def check_dataset_divide_status(dataset, attrset):
    """
    检查数据集在属性集上是否可分\n
    :param dataset: 数据集
    :param attrset: 属性集
    :return:
    """
    status = None
    for i in range(0, len(attrset)):
        if attrset[i] == 1:
            if status is None:
                status = cal_attr_num(dataset, i)
                continue
            if status != cal_attr_num(dataset, i):
                return False
    return True


def tree_generate(dataset, flag, attrset, key='root', father_attr_index=-1, is_log=False):
    """
    递归产生决策树节点\n
    :param dataset: 数据集
    :param flag: 标签集
    :param attrset: 属性集
    :param key: 当前节点关键字
    :param father_attr_index: 父节点属性位置
    :param is_log: 是否输出日志
    :return: 决策树存放于全局列表 tree_node_list
    """
    flag_class = cal_class(flag)
    max_class = cal_max_class(flag=flag, flag_class=flag_class)
    node_number = len(dataset)
    # 生成一个节点
    node = create_node(key=key, attr_index=-1,
                       father_attr_index=father_attr_index,
                       node_class='',
                       node_number=node_number,
                       elements=dataset)

    if flag_class.get(max_class) == node_number:
        # 只有一种类别，标记为该类，作为叶节点
        node[3] = max_class
        tree_node_list.append(node)
        return
    if max(attrset) == 0 or check_dataset_divide_status(dataset, attrset):
        # D 中样本在 A 上取值相同，标记为 D 中样本数最多的类，作为叶节点
        node[3] = max_class
        tree_node_list.append(node)
        return
    # 做最佳划分
    max_gain = 0
    best_index = 0
    for i in range(0, len(attrset)):
        if attrset[i] == 1:
            gain = cal_gain(dataset, flag, i)
            if max_gain < gain:
                max_gain = gain
                best_index = i
    node[1] = best_index
    tree_node_list.append(node)
    if is_log:
        print(f"max gain: {max_gain}, best attr index: {best_index}")
    # 从属性集中去除
    attrset[best_index] = 0
    # 便利属性取值
    attr_values = ['A', 'C', 'G', 'T']
    datasets, flags = divide_dataset(dataset, flag, best_index)
    for attr in attr_values:
        if len(datasets.get(attr)) == 0:
            max_attr_class = cal_max_class(flags.get(attr))
            node = create_node(key=attr, attr_index=-1, father_attr_index=best_index,
                               node_class=max_attr_class, node_number=len(datasets.get(attr)),
                               elements=datasets.get(attr))
            tree_node_list.append(node)
            return
        else:
            tree_generate(dataset=datasets.get(attr),
                          flag=flags.get(attr),
                          attrset=attrset,
                          key=attr,
                          father_attr_index=best_index,
                          is_log=is_log)


def cal_tree_leaves(tree):
    """
    计算树的叶子节点数\n
    :param tree: 树节点列表
    :return:
    """
    leaf_num = 0
    for node in tree:
        if node[0] == 'root':
            continue
        if node[1] == -1:
            leaf_num += 1
    return leaf_num


def cal_max_depth(tree, index=0):
    """
    计算最大树深度\n
    :param tree: 树节点列表
    :param index: 树节点列表中的位置
    :return:
    """
    tree_node = tree[index]
    if tree_node[1] == -1:
        # 到达叶节点，返回深度 1
        return 1
    else:
        # 非叶节点，返回孩子节点深度 +1
        max_depth = 0
        for i in range(index, len(tree)):
            node = tree[i]
            if node[2] == tree_node[1]:
                # 找到孩子
                depth = cal_max_depth(tree, i)
                if depth > max_depth:
                    max_depth = depth
        return max_depth + 1


def remove_zero_node(tree):
    """
    去除零结点\n
    :param tree:
    :return:
    """
    new_tree = []
    for node in tree:
        if node[4] != 0:
            new_tree.append(node)
    return new_tree


def predict_by_tree(data):
    # todo 实现输出
    print()


# 全局决策树节点列表
tree_node_list = []

if __name__ == '__main__':
    # 读取数据集
    # dataset_train, flag_train = load_dataset(dataset_dir='./datasets/',
    #                                          dataset_name='dna.data',
    #                                          is_log=False)
    # dataset_test, flag_test = load_dataset(dataset_dir='./datasets/',
    #                                        dataset_name='dna.test',
    #                                        is_log=False)
    # # 转换处理数据集
    # dataset_train = transfer_dataset(dataset_train)
    # dataset_test = transfer_dataset(dataset_test)
    # # 属性集
    # attr_set = np.ones([1, 60])[0]
    # tree_generate(dataset=dataset_train, flag=flag_train, attrset=attr_set, is_log=True)

    # print(node_list)
    # for node in tree_node_list:
    #     print(node[:5])
    # print(cal_tree_leaves(tree=tree_node_list))

    # tree_node_list = remove_zero_node(tree_node_list)
    # for node in tree_node_list:
    #     print(node[:5])

    # print(cal_tree_leaves(tree=tree_node_list))
    # print(cal_max_depth(tree_node_list, index=0))
    g = Digraph('G', filename='test', format='png')
    g.node('test1', label='1223')
    g.node('test2', label='abcde \n大撒把看吧 \n大大', fontname="Sarasa Mono SC")
    g.edge('test1', 'test2')
    g.view()
