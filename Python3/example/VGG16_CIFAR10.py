# coding=utf8
"""
@author: Yantong Lai
@date: 08/27/2019
"""

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from model.VGG16 import VGG16


########## 1. Preparation ##########
# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Transform
transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR-10 Dataset
train_dataset = torchvision.datasets.CIFAR10(root="../../data",
                                            download=True,
                                            train=True,
                                            transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root="../../data",
                                           download=True,
                                           train=False,
                                           transform=transform)

# DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

# Dataset classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


########## 2. Model ##########
# Create an instance
vgg16 = VGG16()
if device == 'cuda':
    vgg16.cuda()

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(vgg16.parameters(), lr=learning_rate)


########## 3. Train and Test ##########
def train(net, num_epoch):
    """
    It is a function to train the model.
    :param net: defined model instance
    :param num_epoch: number of epochs
    """
    total_step = len(train_loader)
    for epoch in range(num_epoch):
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter's gradient
            optimizer.zero_grad()

            # Forward pass
            output = net(images)
            loss = criterion(output, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

def test(net):
    """
    It is a function to test the model.
    :param net: defined model instance
    """
    net.eval()

    total_correct = 0
    avg_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            avg_loss += criterion(output, labels)

            pred = torch.argmax(output.data, dim=1)
            total_correct += (pred == labels).sum().item()

        avg_loss = avg_loss / len(test_dataset)
        print("Test Avg. Loss: {}, Accuracy: {}%"
              .format(avg_loss, 100 * total_correct / len(test_dataset)))


########## 4. Main Function ##########
def main():
    """
    It is main function.
    """
    # Train
    train(vgg16, num_epochs)
    # Test
    test(vgg16)
    # Save the model checkpoint
    torch.save(vgg16.state_dict(), "LeNet5_CIFAR10.ckpt")
    print("Saved model successfully!\n")


if __name__ == '__main__':

    main()