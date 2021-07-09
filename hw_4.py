# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt


def load_dataset(dataset_path):
    data_frame = pd.read_excel(dataset_path, sheet_name='Data')
    data_class = {}
    dataset_test = []
    label_test = []
    dataset_train = []
    label_train = []
    for i, data in data_frame.iterrows():
        if data_class.get(f"{data[1]}") is None:
            data_class[f"{data[1]}"] = 1
            dataset_test.append(list(data[2:]))
            label_test.append(data[1])
        else:
            if data_class.get(f"{data[1]}") < 2:
                dataset_test.append(list(data[2:]))
                label_test.append(data[1])
                data_class[f"{data[1]}"] += 1
            else:
                dataset_train.append(list(data[2:]))
                label_train.append(data[1])
    columns_name = data_frame.columns.values.tolist()[2:]
    return dataset_train, label_train, dataset_test, label_test, columns_name


# 定义KNN算法分类器函数
# 函数参数包括：(测试数据，训练数据，分类,k值)
def knn_classify(data_test, dataset_train, label_train, k):
    dataset_size = len(dataset_train)
    # 将 测试数据
    diffMat = np.tile(data_test, (dataset_size, 1)) - dataset_train

    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5  # 计算欧式距离
    sortedDistIndicies = distances.argsort()  # 排序并返回index
    # 选择距离最近的k个值
    classCount = {}
    for i in range(k):
        voteIlabel = label_train[sortedDistIndicies[i]]
        # D.get(k[,d]) -> D[k] if k in D, else d. d defaults to None.
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


class KNearestNeighbor(object):
    def __init__(self):
        pass

    # 训练函数
    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    # 预测函数
    def predict(self, X, k=1):
        # 计算L2距离
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))  # 初始化距离函数
        # because(X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train, so
        d1 = -2 * np.dot(X, self.X_train.T)  # shape (num_test, num_train)
        d2 = np.sum(np.square(X), axis=1, keepdims=True)  # shape (num_test, 1)
        d3 = np.sum(np.square(self.X_train), axis=1)  # shape (1, num_train)
        dist = np.sqrt(d1 + d2 + d3)
        # 根据K值，选择最可能属于的类别
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            dist_k_min = np.argsort(dist[i])[:k]  # 最近邻k个实例位置
            y_kclose = self.y_train[dist_k_min]  # 最近邻k个实例对应的标签
            y_pred[i] = np.argmax(np.bincount(y_kclose.tolist()))  # 找出k个标签中从属类别最多的作为预测类别

        return y_pred


# 定义一个生成“训练样本集”的函数，包含特征和分类信息
def createDataSet():
    group = [[1, 1.1], [1, 1], [0, 0], [0, 0.1]]
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def draw_fig(columns_name):
    plt.clf()
    plt.title(f"KNN RESULT ({name_columns[first_index]},{name_columns[second_index]})")
    plt.scatter([i[first_index] for i in train_dataset], [i[second_index] for i in train_dataset], color='red',
                marker='o',
                label='setosa')
    # plt.scatter(X_versicolor[:, 0], X_versicolor[:, 2], color='blue', marker='^', label='versicolor')
    # plt.scatter(X_virginica[:, 0], X_virginica[:, 2], color='green', marker='s', label='virginica')
    plt.xlabel(f'{name_columns[first_index]}')
    plt.ylabel(f'{name_columns[second_index]}')
    plt.legend(loc='upper left')
    # plt.show()
    plt.savefig(f"./hw4/{first_index}_{second_index}.png")


if __name__ == "__main__":
    dataset_file = "./datasets/BreastTissue/BreastTissue.xls"
    train_dataset, train_label, test_dataset, test_label, name_columns = load_dataset(dataset_file)

    for first_index in range(9):
        for second_index in range(9):
            if first_index == second_index:
                continue

    # print(np.tile([12, 13], (2, 3)))
    # group, labels = createDataSet()
    # # 对测试数据[0,0]进行KNN算法分类测试
    # s = knn_classify([0, 0], group, labels, 3)
    # print(s)
