# coding=utf8
"""
@author: Yantong Lai
@date: 08/21/2019
"""

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # C1 (For MNIST, it's (1, 6, 5). However, for CIFAR10, it's (3, 6, 5))
            nn.ReLU(),  # ReLU
            nn.MaxPool2d(2, stride=2),  # S2
            nn.Conv2d(6, 16, 5),  # C3
            nn.ReLU(),  # ReLU
            nn.MaxPool2d(2, stride=2))  # S4

        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), # F5
            nn.ReLU(),
            nn.Linear(120, 84),  # F6
            nn.ReLU(),  # ReLU
            nn.Linear(84, 10),  # F7
            nn.LogSoftmax(dim=-1))  # LogSoftmax

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output