# coding=utf8
"""
@author: Yantong Lai
@date: 08/21/2019
"""

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

from model.LeNet5 import LeNet5

# Device configuaration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Create an LeNet5 instance
net = LeNet5()
# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root="../../data",
                                           train=True,
                                           transform=transforms.Compose([
                                               transforms.Resize((32, 32)),
                                               transforms.ToTensor()]),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.Compose([
                                              transforms.Resize((32, 32)),
                                              transforms.ToTensor()]))

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Train the model
def train(num_epoch):
    """
    It is a functino to train the LeNet5 model
    @param: num_epoch: the number of epochs
    """
    total_step = len(train_loader)
    for epoch in range(num_epoch):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            output = net(images)
            loss = criterion(output, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

def test():
    '''
    It is a function to test the model.
    '''
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        output = net(images)
        avg_loss += criterion(output, labels)
        # _, pred = torch.max(output.data, 1)
        pred = torch.argmax(output.data, dim=1)
        total_correct += (pred == labels).sum().item()

    avg_loss = avg_loss / len(test_dataset)
    print("Test Avg. Loss: {}, Accuracy: {}%"
          .format(avg_loss, 100 * total_correct / len(test_dataset)))

def main():
    """
    Main function
    """
    # Train
    train(num_epochs)
    # Test
    test()
    # Save the model checkpoint
    torch.save(net.state_dict(), "LeNet5.ckpt")
    print("Saved model successfully!\n")


if __name__ == '__main__':

    main()