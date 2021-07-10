# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def load_dataset(dataset_path):
    """
    加载数据集\n
    :param dataset_path: 数据集路径
    :return:
    """
    data_frame = pd.read_excel(dataset_path, sheet_name='Data')
    datasets = []
    labels = []
    for i, data in data_frame.iterrows():
        datasets.append(list(data[2:]))
        labels.append(data[1])
    columns_name = data_frame.columns.values.tolist()[2:]
    return datasets, labels, columns_name


class KMeans(object):
    def __init__(self, k=6, tolerance=0.0001, max_iter=300):
        self.k = k  # 聚簇数量
        self.tolerance = tolerance  # 中心点误差
        self.max_iter = max_iter  # 最大迭代次数
        self.centers = {}  # 簇中心点
        self.clf = {}  # 聚簇分类字典

    def fit(self, train_data):
        self.centers = {}
        # 取前 k 个作为初始均值向量
        for i in range(self.k):
            self.centers[i] = train_data[i]
        # 迭代求解
        for i in range(self.max_iter):
            self.clf = {}
            for j in range(self.k):
                self.clf[j] = []
            for feature in train_data:
                distances = []
                for center in self.centers:
                    # 欧拉距离
                    distances.append(np.linalg.norm(feature - self.centers[center]))
                classification = distances.index(min(distances))
                self.clf[classification].append(feature)

            # 更新均值向量
            prev_centers = dict(self.centers)
            for c in self.clf:
                self.centers[c] = np.average(self.clf[c], axis=0)

            # 检测簇心是否不再变动
            optimized = True
            for center in self.centers:
                org_centers = prev_centers[center]
                cur_centers = self.centers[center]
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance:
                    optimized = False
            # 已达到最优则退出
            if optimized:
                break

    def predict(self, test_data):
        distances = [np.linalg.norm(test_data - self.centers[center]) for center in self.centers]
        index = distances.index(min(distances))
        return index


def draw_fig(first_index, second_index, class_dict, k, columns_name):
    """
    绘制散点图\n
    :param first_index: 属性 1 位置
    :param second_index: 属性 2 位置
    :param class_dict: 类别字典
    :param k: 类别
    :param columns_name: 属性名
    :return:
    """
    color_list = ['b', 'g', 'r', 'c', 'm', 'y']
    marker_list = ['o', '*', '.', 'x', '+', 's']
    plt.clf()
    plt.title(f"K Means RESULT ({columns_name[first_index]},{columns_name[second_index]})")
    for index in range(k):
        dataset = class_dict.get(index)
        plt.scatter([i[0] for i in dataset], [i[1] for i in dataset],
                    color=f"{color_list[index]}",
                    marker=f"{marker_list[index]}",
                    label=index)

    plt.xlabel(f'{columns_name[first_index]}')
    plt.ylabel(f'{columns_name[second_index]}')
    plt.legend()
    # plt.show()
    plt.savefig(f"./hw4/{first_index}_{second_index}.png")


def cal_accuracy(datasets, labels, model):
    class_dict = {}
    for index in range(6):
        class_dict[index] = []
    for index in range(len(labels)):
        label = labels[index]
        if label == 'car':
            dict_index = 0
        elif label == 'fad':
            dict_index = 1
        elif label == 'mas':
            dict_index = 2
        elif label == 'gla':
            dict_index = 3
        elif label == 'con':
            dict_index = 4
        else:
            # label = 'adi'
            dict_index = 5
        class_dict[dict_index].append(model.predict(np.array(datasets[index])))
    accuracy = 0
    for index in range(6):
        data_list = class_dict.get(index)
        # 获取该类别中数量最多的类别个数，即类别正确率
        correct_num = Counter(data_list).most_common(1)[0][1]
        # 加权计算
        accuracy += len(data_list) * (correct_num / len(data_list))
    accuracy = accuracy / len(labels)
    return accuracy


if __name__ == "__main__":
    dataset_file = "./datasets/BreastTissue/BreastTissue.xls"
    datasets, labels, name_columns = load_dataset(dataset_file)

    best_accuracy = 0
    best_first_index = 0
    best_second_index = 0
    for first_index in range(9):
        for second_index in range(9):
            if first_index == second_index:
                continue
            k_means = KMeans(k=6)
            sub_dataset = [[i[first_index], i[second_index]] for i in datasets]
            k_means.fit(np.array(sub_dataset))
            accuracy = cal_accuracy(sub_dataset, labels, k_means)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_first_index = first_index
                best_second_index = second_index
                print(accuracy)
                draw_fig(first_index, second_index, k_means.clf, 6, name_columns)

    print(f"best accuracy: {best_accuracy}, "
          f"using attribute {name_columns[best_first_index]} and {name_columns[best_second_index]}")
