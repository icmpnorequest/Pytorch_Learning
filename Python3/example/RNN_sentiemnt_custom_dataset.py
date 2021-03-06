# coding=utf8
"""
@author: Yantong Lai
@date: 2019.11.9
"""

import torchtext
from torchtext.data import get_tokenizer
from torchtext import data, datasets

import torch
import torch.nn as nn
import torch.optim as optim

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
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


####################################
#          Preparing Data          #
####################################
# 1. data.Field()
TEXT = data.Field(tokenize='spacy', include_lengths=True)
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
class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(RNN, self).__init__()

        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2. RNN layer
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        # 3. Linear layer
        self.fc = nn.Linear(in_features=hidden_dim * 2,
                            out_features=output_dim)

    def forward(self, text, text_lengths):

        # 1. Embedding
        # text = [sent len, batch size]
        embedded = self.embedding(text)

        # 2. Pack sequence
        # embedded = [sent len, batch size, embed size]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)

        # 3. RNN
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # 4. Unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num_directions]
        # output over padding tokens are zero tensors

        # hidden = [num_layers * num_directions, batch size, hid dim]
        # cell = [num_layers * num_directions, batch_size, hid dim]

        # 5. Concat the final forward (hidden[-2, :, :]) and backward (hidden[-1, :, :])
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden = [batch size, hid dim * num_directions]

        # return self.fc(hidden.squeeze(0)).view(-1)
        return self.fc(hidden)

# Parameters
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 2
N_LAYERS = 2
BIDIRECTIONAL = True
DROUPOUT = 0.5
# PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

print("INPUT_DIM = {}".format(INPUT_DIM))
print("OUTPUT_DIM = {}".format(OUTPUT_DIM))
print("TEXT.pad_token = {}".format(TEXT.pad_token))
# print("PAD_IDX = {}".format(PAD_IDX))

# Create an RNN instance
model = RNN(vocab_size=INPUT_DIM,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            n_layers=N_LAYERS,
            bidirectional=BIDIRECTIONAL,
            dropout=DROUPOUT)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("The model has {} trainable parameters".format(count_parameters(model)))


####################################
#          Train the Model         #
####################################
# Loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model = model.to(device)


########## Train ##########
NUM_EPOCHS = 10
total_step = len(train_iter)
for epoch in range(NUM_EPOCHS):
    total_loss = []
    for i, batch in enumerate(train_iter):

        text, text_lengths = batch.text
        y = batch.labels
        # print("y = {}".format(y))

        # Forward pass
        y_pred = model(text, text_lengths)
        # print("y_pred = {}".format(y_pred))
        loss = criterion(y_pred, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item()))

    print("total_loss = {}\n".format(total_loss))


########## Test ##########
model.eval()
total_correct = 0
avg_loss = 0.0
for i, batch in enumerate(test_iter):
    text, text_lengths = batch.text

    y = batch.labels

    # Forward pass
    y_pred = model(text, text_lengths)
    loss = criterion(y_pred, y)
    avg_loss += loss.item()

    # _, pred = torch.max(output.data, 1)
    pred = torch.argmax(y_pred.data, dim=1)
    total_correct += (pred == y).sum().item()

avg_loss = avg_loss / len(test_data)
print("Test Avg. Loss: {}, Accuracy: {}%"
      .format(avg_loss, 100 * total_correct / len(test_data)))
