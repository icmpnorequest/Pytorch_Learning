# coding = utf8
"""
@author: Yantong Lai
@date: 2019.10.31
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

lstm = nn.LSTM(3, 3)        # input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5
print("inputs = {}".format(inputs))

# Initialize the hidden state
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
print("hidden = {}\n".format(hidden))

for i in inputs:
    out, hidden = lstm(i.view(1, 1, -1), hidden)

inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))
out, hidden = lstm(inputs, hidden)
print("out = {}".format(out))
print("hidden = {}".format(hidden))




