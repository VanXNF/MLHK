# -*- coding: utf-8 -*-
import math
import random
import numpy as np


def rand(a, b):
    return (b - a) * random.random() + a


def init_matrix(m, n, is_zero=True):
    """
    创建 m*n 的矩阵
    :param m: 矩阵行数
    :param n: 矩阵列数
    :param is_zero: 是否为 0
    :return:
    """
    if is_zero:
        return np.zeros([m, n])
    else:
        return np.ones([m, n])


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


if __name__ == "__main__":
    a = init_matrix(3, 4, is_zero=False)
    a[1][1] = 222
    print(a)
