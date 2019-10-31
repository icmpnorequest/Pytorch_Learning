# coding = utf8
"""
@author: Yantong Lai
@date: 2019.10.31
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("The dog ate the apple".split(),
     ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(),
     ["NN", "V", "DET", "NN"])
]

word_to_idx = {}
tag_to_idx = {}
for sent, tags in training_data:
    # print("sent = {}".format(sent))
    # print("tags = {}".format(tags))
    for word in sent:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
    for tag in tags:
        if tag not in tag_to_idx:
            tag_to_idx[tag] = len(tag_to_idx)

print("word_to_idx = {}".format(word_to_idx))
print("tag_to_idx = {}".format(tag_to_idx))


EMBEDDING_DIM = 6
HIDDEN_DIM = 6
NUM_EPOCHS = 3


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(in_features=hidden_dim, out_features=tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = LSTMTagger(embedding_dim=EMBEDDING_DIM,
                   hidden_dim=HIDDEN_DIM,
                   vocab_size=len(word_to_idx),
                   tagset_size=len(tag_to_idx))

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


# See what the scores are before training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_idx)
    tag_scores = model(inputs)
    print(tag_scores)
    print(tag_scores.size())


for epoch in range(NUM_EPOCHS):
    for sentence, tags in training_data:
        model.zero_grad()

        sentence_in = prepare_sequence(sentence, word_to_idx)
        targets = prepare_sequence(tags, tag_to_idx)

        tag_scores = model(sentence_in)

        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()


# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_idx)
    tag_scores = model(inputs)
    print(tag_scores)





