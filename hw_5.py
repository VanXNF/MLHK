# -*- coding: utf-8 -*-

import os
import math
import torch
from hw_5_dataset import FeatherImg
from hw_5_model import FeatherNet
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from hw_5_tools import write_list_to_txt
import matplotlib.pyplot as plt

# 加载数据集
root_dir = './feather_dataset'
train_dataset = FeatherImg(root_dir, train=True, transform=transforms.ToTensor())
test_dataset = FeatherImg(root_dir, train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True, num_workers=0)

# 新建模型
print('创建羽毛模型')
feather_net = FeatherNet()
feather_net.cuda()
weight_path = root_dir + '/feather.pkl'
if os.path.exists(weight_path):
    print('加载上次训练参数')
    feather_net.load_state_dict(torch.load(weight_path))
# 损失函数
criterion = nn.CrossEntropyLoss().cuda()
# 优化器
optimizer = optim.SGD(feather_net.parameters(), lr=0.001, momentum=0.9)
# 类别
classes = ('1', '2', '3', '4', '56')

# 损失及精度数据收集
loss_list = []
y1 = []
y2 = []
y3 = []
y4 = []
y5 = []
y_list = [y1, y2, y3, y4, y5]
epochs = 800

pbar = tqdm(total=epochs)
print(f'Epoch {epochs}')
for epoch in range(epochs):

    running_loss = 0.0
    for step, (inputs, labels) in enumerate(train_loader, 0):

        inputs = inputs.cuda()
        labels = labels.cuda()
        # 梯度清零
        optimizer.zero_grad()

        out = feather_net(inputs)
        loss = criterion(out, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #         print(f'step {step+1}')
        # 训练集 4674/32 -> 147 steps
        max_step = math.ceil(len(train_loader.dataset) / train_loader.batch_size)
        if (step + 1) % max_step == 0:
            print(f'Epoch {epoch + 1} : {running_loss / max_step}', end='\r')
            if (epoch + 1) % 10 == 0:
                loss_list.append(running_loss / max_step)
                class_correct = list(0. for i in range(5))
                class_total = list(0. for i in range(5))
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.cuda(), labels.cuda()
                        outputs = feather_net(images)
                        _, predicted = torch.max(outputs, 1)
                        c = (predicted == labels).squeeze()
                        for i in range(len(labels)):
                            label = labels[i]
                            class_correct[label] += c[i].item()
                            class_total[label] += 1
                for i in range(5):
                    print(f'测试集准确率 类别 {classes[i]:5s} : {(100 * class_correct[i] / class_total[i]):2.2f} %')
                    y_list[i].append(100 * class_correct[i] / class_total[i])

                torch.save(feather_net.state_dict(), weight_path)
                print(f'epoch {epoch + 1}: 保存模型参数成功', end='\r')
                pbar.update(10)
            running_loss = 0.0

pbar.close()
print('训练完成')

torch.save(feather_net.state_dict(), weight_path)
print('保存模型参数成功')

write_list_to_txt(loss_list, root_dir + '/loss_log.txt')
write_list_to_txt(y1, root_dir + '/acc_class_1.txt')
write_list_to_txt(y2, root_dir + '/acc_class_2.txt')
write_list_to_txt(y3, root_dir + '/acc_class_3.txt')
write_list_to_txt(y4, root_dir + '/acc_class_4.txt')
write_list_to_txt(y5, root_dir + '/acc_class_56.txt')
print('训练记录保存成功')
# 直接执行脚本时不输出图像
# x_axix = range(10, epochs+1, 10)

# plt.plot(x_axix, loss_list, 'o-')
# plt.title('train loss per 10 epoches')
# plt.ylabel('loss')
# plt.xlabel('iteration times')
# plt.savefig(root_dir + "/loss.jpg")
# plt.show()

# plt.title('test accuracy per 10 epoches')
# plt.plot(x_axix, y1, color='black', label="class 1")
# plt.plot(x_axix, y2, color='green', label='class 2')
# plt.plot(x_axix, y3, color='red', label='class 3')
# plt.plot(x_axix, y4, color='skyblue', label='class 4')
# plt.plot(x_axix, y5, color='blue', label='class 56')
# plt.legend(loc="lower right")
# plt.xlabel('iteration times')
# plt.ylabel('rate %')
# plt.savefig(root_dir + "/accuracy.jpg")
# plt.show()
