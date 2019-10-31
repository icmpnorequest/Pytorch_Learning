# coding=utf8
"""
@author: Yantong Lai
@date: 08/22/2019
"""

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 5
batch_size = 1000
learning_rate = 0.001

# Classes
classes = ('plane', 'car', 'bird','cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# CIFAR10 dataset
# Normalize the CIFAR10 dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root="../../data",
                                             download=True,
                                             train=True,
                                             transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)

test_dataset = torchvision.datasets.CIFAR10(root="../../data",
                                            download=True,
                                            train=False,
                                            transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)


class LeNet5_cifar(nn.Module):
    def __init__(self):
        super(LeNet5_cifar, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(3, 6, 5),  # C1
            nn.ReLU(),  # ReLU
            nn.MaxPool2d(2, stride=2),  # S2
            nn.Conv2d(6, 16, 5),  # C3
            nn.ReLU(),  # ReLU
            nn.MaxPool2d(2, stride=2))  # S4

        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
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


# Create an LeNet5 instance
net = LeNet5_cifar()
# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Train the model
def train(num_epoch):
    """
    It is a function to train the LeNet5 model
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

            if (i + 1) % 2000 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

def test():
    '''
    It is a function to test the model.
    '''
    net.eval()

    total_correct = 0
    avg_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            avg_loss += criterion(output, labels)

            # pred = torch.argmax(output.data, dim=1)
            # print("pred = ", pred)

            _, pred = torch.max(output.data, dim=1)
            print("_ = ",  _)                           # probability
            print("pred = ", pred)                      # index of the probability

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
    torch.save(net.state_dict(), "LeNet5_CIFAR10.ckpt")
    print("Saved model successfully!\n")


if __name__ == '__main__':

    main()