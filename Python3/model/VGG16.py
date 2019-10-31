# coding=utf8
"""
@author: Yantong Lai
@date: 08/27/2019
"""

import torch.nn as nn


class VGG16(nn.Module):
    """
    Input - 224x224x3
    Conv1 - 224x224@64
    Conv2 - 224x224@64
    MaxPool1 - 112x112@64
    Conv3 - 112x112@128
    Conv4 - 112x112x128
    MaxPool2 - 56x56x128
    Conv5 - 56x56x256
    Conv6 - 56x56x256
    Conv7 - 56x56x256
    MaxPool3 - 28x28x256
    Conv8 - 28x28x512
    Conv9 - 28x28@512
    Conv10 - 28x28x512
    MaxPool4 - 14x14@512
    Conv11 - 14x14@512
    Conv12 - 14x14@512
    Conv13 - 14x14@512
    MaxPool5 - 7x7@512
    FC1 - 1x1@4096
    FC2 - 1x1@4096
    FC# - 1x1@1000
    """

    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()

        self.convnet = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            # Batch Normalization, to avoid gradient disapperance
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Conv2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # MaxPool
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Conv4
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # MaxPool
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv5
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Conv6
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Conv7
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # MaxPool
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv8
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Conv9
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Conv10
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # MaxPool
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv11
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Conv12
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Conv13
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # MaxPool
            nn.MaxPool2d(kernel_size=2, stride=2),

            # AvgPool
            nn.AvgPool2d(kernel_size=1, stride=1))

        self.fc = nn.Sequential(
            # FC1
            nn.Linear(in_features=512, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),

            # FC2
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),

            # FC3
            nn.Linear(in_features=4096, out_features=num_classes))

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output