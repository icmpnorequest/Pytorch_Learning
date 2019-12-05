# coding=utf8
"""
@author: Yantong Lai
@date: 12/5/2019
"""

import torchtext
from torchtext.data import get_tokenizer
from torchtext import data, datasets

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import spacy
spacy.load("en_core_web_sm")

import random

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Dataset path
dataset_path = "../../data/aclImdb/"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


####################################
#         Hyper-parameters         #
####################################
BATCH_SIZE = 256
LEARNING_RATE = 1e-3


####################################
#          Preparing Data          #
####################################
# 1. data.Field()
TEXT = data.Field('spacy')
LABELS = data.LabelField()

# 2. data.TabularDataset
train_data, test_data = data.TabularDataset.splits(path=dataset_path,
                                                   train="train.tsv",
                                                   test="test.tsv",
                                                   fields=[('labels', LABELS), ('text', TEXT)],
                                                   format="tsv")

# train_data, test_data = datasets.IMDB.splits(TEXT, LABELS)

print("Number of train_data = {}".format(len(train_data)))
print("Number of test_data = {}".format(len(test_data)))

print("vars(train_data[0]) = {}\n".format(vars(train_data[0])))

# 3. Split train_data to train_data, valid_data
train_data, valid_data = train_data.split(random_state=random.seed(SEED))
print("Number of train_data = {}".format(len(train_data)))
print("Number of valid_data = {}".format(len(valid_data)))
print("Number of test_data = {}\n".format(len(test_data)))

# 4. data.BucketIterator
train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                               batch_size=BATCH_SIZE,
                                                               device=device,
                                                               sort_key=lambda x: len(x.text))
# 5. Build vocab
# TEXT.build_vocab(train_data)
# unk_init=torch.Tensor.normal_)
# LABELS.build_vocab(train_data)
# print("vars(train_data[0]) = ", vars(train_data[0]))

# 5.1 (Optional) If build vocab with pre-trained word embedding vectors
TEXT.build_vocab(train_data, vectors="glove.6B.100d")
LABELS.build_vocab(train_data)
print("vars(train_data[0]) = ", vars(train_data[0]))


####################################
#          Build the Model         #
####################################
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.conv_0 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[0], embedding_dim))

        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[1], embedding_dim))

        self.conv_2 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[2], embedding_dim))

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]

        text = text.permute(1, 0)

        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        # torch.nn.functional.max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

# Parameters
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 200
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = len(LABELS.vocab)
DROPOUT = 0.5

# Create an instance
model = TextCNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)


####################################
#          Train the Model         #
####################################
# criterion = nn.BCEWithLogitsLoss().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model = model.to(device)

########## Train ##########
NUM_EPOCHS = 10
total_step = len(train_iter)
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = []
    train_total_correct = 0

    for i, batch in enumerate(train_iter):

        text = batch.text
        y = batch.labels

        # Forward pass
        # y_pred = model(text).squeeze(1).float()
        y_pred = model(text).squeeze(1)

        loss = criterion(y_pred, y)

        pred = torch.argmax(y_pred.data, dim=1)
        train_total_correct += (pred == y).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))

    print("total_loss = {}".format(total_loss))
    print("total_accuracy = {:.4f}%\n".format(100 * train_total_correct / len(train_data)))


########## Evaluation ##########
model.eval()
total_correct = 0
total_loss = 0.0

for i, batch in enumerate(valid_iter):
    text = batch.text
    y = batch.labels

    # Forward pass
    # y_pred = model(text).squeeze(1).float()
    y_pred = model(text).squeeze(1)

    loss = criterion(y_pred, y)

    pred = torch.argmax(y_pred.data, dim=1)
    total_correct += (pred == y).sum().item()

    total_loss += loss.item()

    if (i + 1) % 10 == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))

avg_loss = total_loss / len(valid_data)
print("Test Avg. Loss: {:.4f}, Accuracy: {:.4f}%"
      .format(avg_loss, 100 * total_correct / len(valid_data)))
