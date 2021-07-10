# -*- coding: utf-8 -*-
import torch.nn as nn


class FeatherNet(nn.Module):
    def __init__(self):
        super(FeatherNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),  # (1,32,80) --> (16,32,80)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # (16,32,80) --> (16,16,40)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),  # (16,16,40) --> (32,18,42)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)  # (32,18,42) --> (32,6,14)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(  # (32,6,14) --> (64,6,14)
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # (64,6,14) --> (64,3,7)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 7, 100),
            nn.ReLU()
        )
        self.output = nn.Linear(100, 5)

    def forward(self, x):
        out = self.conv1(x)  # (Batch,1,32,80) --> (Batch,16,16,40)
        out = self.conv2(out)  # (Batch,16,16,40) --> (Batch,32,6,14)
        out = self.conv3(out)  # (Batch,32,6,14) --> (Batch,64,3,7)
        out = out.view(out.size(0), -1)  # (Batch,64,3,7) --> (Batch, 64*3*7)
        out = self.fc(out)
        out = self.output(out)
        return out
