# -*- coding: utf-8 -*-
import math
import random
import os
import numpy as np
from matplotlib import pyplot as plt
import pylab as pl

random.seed(0)


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
            data_line = data_line.replace('\n', '')[:-1]
            data_line = data_line.split(' ')
            list_data = []
            for i in data_line:
                list_data.append(int(i))
            data.append(list_data[:-1])
            list_flag = [0, 0, 0]
            flag_class = int(list_data[-1])
            list_flag[flag_class - 1] = 1
            flag.append(list_flag)
            if is_log:
                print(f"{dataset_name}: class: {list_data[-1]}, len: {len(list_data)}, data: {list_data}")
    return data, flag


def transfer_str_to_list(list_str, dims=1):
    """
    将以字符串形式存储的列表转化为列表
    :param list_str: 字符串
    :param dims: 列表维度 1或2
    :return:
    """
    if dims == 1:
        target_list = []
        list_str = list_str.replace('\n', '').replace('[', '').replace(']', '').replace(' ', '').split(',')
        for i in list_str:
            target_list.append(float(i))
        return target_list
    elif dims == 2:
        target_list = []
        list_str = list_str.replace('\n', '').replace(' ', '')[1:-1].replace('],[', ']#[').split('#')
        for i in list_str:
            target_list.append(transfer_str_to_list(i))
        return target_list


def rand(a, b):
    """
    用于随机初始化数值
    :param a:
    :param b:
    :return:
    """
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    """
    创建一个 m*n 的矩阵
    :param m: 行数
    :param n: 列数
    :param fill: 填充值
    :return:
    """
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    """
    sigmoid 激活函数
    :param x:
    :return:
    """
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(fx):
    """
    sigmoid 函数导数
    :param fx: 原函数
    :return:
    """
    return fx * (1 - fx)


def draw_error_log(log_path):
    """
    绘制错误率曲线\n
    :param log_file_path: 日志路径
    :return:
    """
    y = []
    with open(log_file_path, 'r') as log:
        for line in log.readlines():
            y.append(float(line))
    x = range(len(y))

    pl.plot(x, y, 'r', label='error rate')

    pl.title('Train Error Rate')
    pl.xlabel('Epoch')
    pl.ylabel('Error Rate')

    pl.xlim(0.0, 1000.0)
    pl.ylim(0.0, 40.0)
    pl.legend()
    pl.show()


class BPNetwork:
    def __init__(self):
        self.input_neural_num = 0
        self.hidden_neural_num = 0
        self.output_neural_num = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []
        self.model_path = None
        self.epoch = 0

    def setup(self, ni, nh, no, model_path=None):
        """
        初始化 BP 网络
        :param ni: 输入层神经元数量
        :param nh: 隐藏层神经元数量
        :param no: 输出层神经元数量
        :param model_path: 模型存放位置
        :return:
        """
        self.input_neural_num = ni + 1  # 多一个偏置神经元
        self.hidden_neural_num = nh
        self.output_neural_num = no
        self.epoch = 0
        if model_path is None:
            self.model_path = f"./{self.input_neural_num}_{self.hidden_neural_num}_{self.output_neural_num}.model"
        else:
            self.model_path = model_path
        # 初始化神经元列表
        self.input_cells = [1.0] * self.input_neural_num
        self.hidden_cells = [1.0] * self.hidden_neural_num
        self.output_cells = [1.0] * self.output_neural_num
        # 初始化权值和阈值矩阵
        self.input_weights = make_matrix(self.input_neural_num, self.hidden_neural_num)
        self.output_weights = make_matrix(self.hidden_neural_num, self.output_neural_num)
        # 随机初始化权值和阈值
        for i in range(self.input_neural_num):
            for h in range(self.hidden_neural_num):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_neural_num):
            for o in range(self.output_neural_num):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        # 初始化矫正矩阵
        self.input_correction = make_matrix(self.input_neural_num, self.hidden_neural_num)
        self.output_correction = make_matrix(self.hidden_neural_num, self.output_neural_num)

    def predict(self, inputs):
        """
        前馈过程
        :param inputs: 输入特征列表
        :return:
        """
        # 激活输入层
        for i in range(self.input_neural_num - 1):
            self.input_cells[i] = inputs[i]
        # 激活隐藏层
        for j in range(self.hidden_neural_num):
            total = 0.0
            for i in range(self.input_neural_num):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # 激活输出层
        for k in range(self.output_neural_num):
            total = 0.0
            for j in range(self.hidden_neural_num):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    def back_propagate(self, case, label, learn_rate, correct_rate):
        """
        反向传播过程
        :param case: 样本
        :param label: 标签
        :param learn_rate: 学习率
        :param correct_rate: 矫正率
        :return:
        """
        self.predict(case)
        # 计算输出层误差
        output_deltas = [0.0] * self.output_neural_num
        for o in range(self.output_neural_num):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        # 计算隐藏层误差
        hidden_deltas = [0.0] * self.hidden_neural_num
        for h in range(self.hidden_neural_num):
            error = 0.0
            for o in range(self.output_neural_num):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error
        # 更新输出层权值
        for h in range(self.hidden_neural_num):
            for o in range(self.output_neural_num):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn_rate * change + correct_rate * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # 更新输入层权值
        for i in range(self.input_neural_num):
            for h in range(self.hidden_neural_num):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn_rate * change + correct_rate * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # 计算全局误差
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    def train(self, cases, labels, limit=10000, learn_rate=0.05, correct_rate=0.1, log_file_path=None):
        """
        训练
        :param cases: 样本列表
        :param labels: 标签列表
        :param limit: 最大迭代次数
        :param learn_rate: 学习率
        :param correct_rate: 矫正率
        :param log_file_path: 日志文件路径
        :return:
        """
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn_rate, correct_rate)
            if log_file_path is not None:
                with open(log_file_path, 'a') as log:
                    log.write(f"{error}\n")
            print(f"epoch: {j + self.epoch}, error: {error}")
            if j % 10 == 0 or j == limit - 1:
                with open(self.model_path, "w") as model:
                    model.write(f'{self.input_neural_num},{self.hidden_neural_num},{self.output_neural_num},{j}\n')
                    model.write(str(self.input_cells) + '\n')
                    model.write(str(self.hidden_cells) + '\n')
                    model.write(str(self.output_cells) + '\n')
                    model.write(str(self.input_weights) + '\n')
                    model.write(str(self.output_weights) + '\n')
                    model.write(str(self.input_correction) + '\n')
                    model.write(str(self.output_correction) + '\n')

    def test(self, test_cases, test_labels, is_log=False):
        """
        测试
        :return:
        """
        error_count = 0
        for i in range(len(test_cases)):
            predict_list = self.predict(test_cases[i])
            if predict_list.index(max(predict_list)) != test_labels[i].index(max(test_labels[i])):
                error_count += 1
            if is_log:
                print(f"predict: {predict_list}, label: {test_labels[i]}")
        print(f"error rate: {error_count}/{len(test_cases)}, {error_count / len(test_cases)}")

    def load_model(self, model_path=None):
        """
        加载模型参数
        :param model_path: 模型路径
        :return:
        """
        if model_path is None:
            model_path = self.model_path
        if os.path.exists(model_path):
            with open(model_path, 'r') as model:
                neural_nums = model.readline()
                neural_nums = neural_nums.split(',')
                self.input_neural_num = int(neural_nums[0])
                self.hidden_neural_num = int(neural_nums[1])
                self.output_neural_num = int(neural_nums[2])
                self.epoch = int(neural_nums[3])
                input_cells = model.readline()
                self.input_cells = transfer_str_to_list(input_cells)
                hidden_cells = model.readline()
                self.hidden_cells = transfer_str_to_list(hidden_cells)
                output_cells = model.readline()
                self.output_cells = transfer_str_to_list(output_cells)
                input_weights = model.readline()
                self.input_weights = transfer_str_to_list(input_weights, 2)
                output_weights = model.readline()
                self.output_weights = transfer_str_to_list(output_weights, 2)
                input_correction = model.readline()
                self.input_correction = transfer_str_to_list(input_correction, 2)
                output_correction = model.readline()
                self.output_correction = transfer_str_to_list(output_correction, 2)
                self.model_path = model_path
            print(f"load model success")
            return True
        else:
            print(f"load model failed")
            return False


if __name__ == "__main__":
    # 读取数据集
    dataset_train, flag_train = load_dataset(dataset_dir='datasets/',
                                             dataset_name='dna.data',
                                             is_log=False)
    dataset_test, flag_test = load_dataset(dataset_dir='datasets/',
                                           dataset_name='dna.test',
                                           is_log=False)
    model_path = "./hw2/180_15_3.model"
    log_file_path = './hw2/180_15_3.log'
    bp = BPNetwork()
    # 训练
    bp.setup(ni=180, nh=15, no=3, model_path=model_path)
    bp.train(cases=dataset_train,
             labels=flag_train,
             limit=1000,
             learn_rate=0.05,
             correct_rate=0.1,
             log_file_path=log_file_path)
    # 绘制错误率曲线
    draw_error_log(log_path=log_file_path)
    # 测试
    bp.load_model(model_path=f"./hw2/180_15_3.model")
    bp.test(dataset_test, flag_test)
