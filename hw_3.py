# -*- coding: utf-8 -*-

from scipy.signal import convolve2d
from skimage.measure import block_reduce
from hw_3_mnist import *
import matplotlib.pyplot as plt


class LeNet(object):
    # The network is like:
    #    conv1 -> pool1 -> conv2 -> pool2 -> fc1 -> relu -> fc2 -> relu -> softmax
    # l0      l1       l2       l3        l4     l5      l6     l7      l8        l9
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.l0 = None
        # 6 convolution kernel, each has 1 * 5 * 5 size
        self.conv1 = xavier_init(6, 1, 5, 5)
        self.l1 = None
        # the size for mean pool is 2 * 2, stride = 2
        self.pool1 = [2, 2]
        self.l2 = None
        # 16 convolution kernel, each has 6 * 5 * 5 size
        self.conv2 = xavier_init(16, 6, 5, 5)
        self.l3 = None
        # the size for mean pool is 2 * 2, stride = 2
        self.pool2 = [2, 2]
        self.l4 = None
        # fully connected layer 256 -> 200
        self.fc1 = xavier_init(256, 200, fc=True)
        self.l5 = None
        self.l6 = None
        # fully connected layer 200 -> 10
        self.fc2 = xavier_init(200, 10, fc=True)
        self.l7 = None
        self.l8 = None
        self.l9 = None

    def forward_prop(self, input_data):
        self.l0 = np.expand_dims(input_data, axis=1) / 255  # (batch_sz, 1, 28, 28)
        self.l1 = self.convolution(self.l0, self.conv1)  # (batch_sz, 6, 24, 24)
        self.l2 = self.mean_pool(self.l1, self.pool1)  # (batch_sz, 6, 12, 12)
        self.l3 = self.convolution(self.l2, self.conv2)  # (batch_sz, 16, 8, 8)
        self.l4 = self.mean_pool(self.l3, self.pool2)  # (batch_sz, 16, 4, 4)
        self.l5 = self.fully_connect(self.l4, self.fc1)  # (batch_sz, 200)
        self.l6 = self.relu(self.l5)  # (batch_sz, 200)
        self.l7 = self.fully_connect(self.l6, self.fc2)  # (batch_sz, 10)
        self.l8 = self.relu(self.l7)  # (batch_sz, 10)
        self.l9 = self.softmax(self.l8)  # (batch_sz, 10)
        return self.l9

    def backward_prop(self, softmax_output, output_label):
        l8_delta = (output_label - softmax_output) / softmax_output.shape[0]
        l7_delta = self.relu(self.l8, l8_delta, derivative=True)  # (batch_sz, 10)
        l6_delta, self.fc2 = self.fully_connect(self.l6, self.fc2, l7_delta, derivative=True)  # (batch_sz, 200)
        l5_delta = self.relu(self.l6, l6_delta, derivative=True)  # (batch_sz, 200)
        l4_delta, self.fc1 = self.fully_connect(self.l4, self.fc1, l5_delta, derivative=True)  # (batch_sz, 16, 4, 4)
        l3_delta = self.mean_pool(self.l3, self.pool2, l4_delta, derivative=True)  # (batch_sz, 16, 8, 8)
        l2_delta, self.conv2 = self.convolution(self.l2, self.conv2, l3_delta, derivative=True)  # (batch_sz, 6, 12, 12)
        l1_delta = self.mean_pool(self.l1, self.pool1, l2_delta, derivative=True)  # (batch_sz, 6, 24, 24)
        l0_delta, self.conv1 = self.convolution(self.l0, self.conv1, l1_delta, derivative=True)  # (batch_sz, 1, 28, 28)

    def convolution(self, input_map, kernel, front_delta=None, derivative=False):
        num, channel, width, height = input_map.shape
        kernel_num, kernel_channel, kernel_width, kernel_height = kernel.shape
        if not derivative:
            feature_map = np.zeros((num, kernel_num, width - kernel_width + 1, height - kernel_height + 1))
            for imgId in range(num):
                for kId in range(kernel_num):
                    for cId in range(channel):
                        feature_map[imgId][kId] += \
                            convolve2d(input_map[imgId][cId], kernel[kId, cId, :, :], mode='valid')
            return feature_map
        else:
            # front->back (propagate loss)
            back_delta = np.zeros((num, channel, width, height))
            kernel_gradient = np.zeros((kernel_num, kernel_channel, kernel_width, kernel_height))
            padded_front_delta = np.pad(front_delta,
                                        [(0, 0), (0, 0), (kernel_width - 1, kernel_height - 1),
                                         (kernel_width - 1, kernel_height - 1)],
                                        mode='constant',
                                        constant_values=0)
            for imgId in range(num):
                for cId in range(channel):
                    for kId in range(kernel_num):
                        back_delta[imgId][cId] += convolve2d(padded_front_delta[imgId][kId],
                                                             kernel[kId, cId, ::-1, ::-1], mode='valid')
                        kernel_gradient[kId][cId] += convolve2d(front_delta[imgId][kId],
                                                                input_map[imgId, cId, ::-1, ::-1], mode='valid')
            # update weights
            kernel += self.learning_rate * kernel_gradient
            return back_delta, kernel

    def mean_pool(self, input_map, pool, front_delta=None, derivative=False):
        # num, channel, width, height = input_map.shape
        pool_num, pool_height = tuple(pool)
        if not derivative:
            # feature_map = np.zeros((num, channel, width // pool_num, height // pool_height))
            feature_map = block_reduce(input_map, tuple((1, 1, pool_num, pool_height)), func=np.mean)
            return feature_map
        else:
            # front->back (propagate loss)
            # back_delta = np.zeros((num, channel, width, height))
            back_delta = front_delta.repeat(pool_num, axis=2).repeat(pool_height, axis=3)
            back_delta /= (pool_num * pool_height)
            return back_delta

    def fully_connect(self, input_data, fc, front_delta=None, derivative=False):
        n = input_data.shape[0]
        if not derivative:
            output_data = np.dot(input_data.reshape(n, -1), fc)
            return output_data
        else:
            # front->back (propagate loss)
            back_delta = np.dot(front_delta, fc.T).reshape(input_data.shape)
            # update weights
            fc += self.learning_rate * np.dot(input_data.reshape(n, -1).T, front_delta)
            return back_delta, fc

    def relu(self, x, front_delta=None, derivative=False):
        if derivative:
            # propagate loss
            back_delta = front_delta * 1. * (x > 0)
            return back_delta
        else:
            return x * (x > 0)

    def softmax(self, x):
        y = list()
        for t in x:
            e_t = np.exp(t - np.max(t))
            y.append(e_t / e_t.sum())
        return np.array(y)


def xavier_init(c1, c2, w=1, h=1, fc=False):
    fan_1 = c2 * w * h
    fan_2 = c1 * w * h
    ratio = np.sqrt(6.0 / (fan_1 + fan_2))
    params = ratio * (2 * np.random.random((c1, c2, w, h)) - 1)
    if fc:
        params = params.reshape(c1, c2)
    return params


def convert_to_one_hot(labels):
    one_hot_labels = np.zeros((labels.size, labels.max() + 1))
    one_hot_labels[np.arange(labels.size), labels] = 1
    return one_hot_labels


def shuffle_dataset(data, label):
    n = data.shape[0]
    index = np.random.permutation(n)
    x = data[index, :, :]
    y = label[index, :]
    return x, y


def draw_acc_loss(log_file):
    """

    :param log_file:
    :return:
    """
    list_accuracy = []
    list_loss = []
    with open(log_file, 'r') as log:
        for line in log.readlines():
            line.replace('\n', '').split(',')
            list_accuracy.append(float(line[0]))
            list_loss.append(float(line[1]) / 150)
    x = [i * 100 for i in range(len(accuracy_list))]
    plt.rcParams['figure.figsize'] = (10, 10)  # 图像显示大小
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['lines.linewidth'] = 2  # 设置曲线线条宽度
    plt.plot(x, accuracy_list, 'r', label='training accuracy')
    plt.plot(x, loss_list, 'b', label='loss/150')

    plt.title('LeNet Training Result', fontsize=30)
    plt.xlabel('Epoch')
    plt.ylabel('Rate')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_imgs = load_train_images()
    train_labs = load_train_labels().astype(int)
    log_file = f"./hw3/lenet.log"
    pic_file = f"./hw3/result.png"

    data_size = train_imgs.shape[0]
    batch_sz = 64
    lr = 0.01
    max_iter = 50000
    iter_mod = int(data_size / batch_sz)
    train_labs = convert_to_one_hot(train_labs)
    le_net = LeNet(learning_rate=lr)
    # 设置绘图相关参数
    plt.ion()
    plt.rcParams['figure.figsize'] = (10, 10)  # 图像显示大小
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['lines.linewidth'] = 2  # 设置曲线线条宽度

    accuracy_list = []
    loss_list = []
    x = []

    for iters in range(max_iter):
        # starting index and ending index for input data
        st_idx = (iters % iter_mod) * batch_sz
        # shuffle the dataset
        if st_idx == 0:
            train_imgs, train_labs = shuffle_dataset(train_imgs, train_labs)
        data_input = train_imgs[st_idx: st_idx + batch_sz]
        label_output = train_labs[st_idx: st_idx + batch_sz]
        softmax_output = le_net.forward_prop(data_input)
        if iters % 100 == 0:
            # 计算精度
            correct_list = [int(np.argmax(softmax_output[i]) == np.argmax(label_output[i])) for i in range(batch_sz)]
            accuracy = float(np.array(correct_list).sum()) / batch_sz
            # 计算损失
            correct_prob = [softmax_output[i][np.argmax(label_output[i])] for i in range(batch_sz)]
            correct_prob = list(filter(lambda x: x > 0, correct_prob))
            loss = -1.0 * np.sum(np.log(correct_prob))
            print(f"The {iters} iters result:")
            print(f"The accuracy is {accuracy} The loss is {loss}")
            with open(log_file, 'a') as log:
                log.write(f"{accuracy},{loss}\n")
            # 绘图
            plt.clf()  # 清除刷新前的图表，防止数据量过大消耗内存
            plt.title("LeNet Training Result", fontsize=30)
            accuracy_list.append(accuracy)
            loss_list.append(loss / 150)
            x.append(iters)
            plt.plot(x, accuracy_list, 'r', label='training accuracy')
            plt.plot(x, loss_list, 'b', label='loss/150')
            plt.xlabel('Epoch')
            plt.ylabel('Rate')
            plt.legend()
            plt.pause(0.4)
        le_net.backward_prop(softmax_output, label_output)
    plt.savefig(pic_file)
    plt.ioff()
    plt.show()
