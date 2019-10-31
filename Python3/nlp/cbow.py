# coding = utf8
"""
@author: Yantong Lai
@date: 2019.10.31
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(1)

# Definitions
CONTEXT_SIZE = 2        # 2 words to the left, 2 words to the right
EMBEDDING_DIM = 100

# Hyper-parameters
learning_rate = 0.001
num_epochs = 10


raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
vocab_size = len(vocab)

word_to_idx = {word:i for i, word in enumerate(vocab)}
print("word_to_idx = {}\n".format(word_to_idx))

data = []
for i in range(2, len(raw_text) - 2):

    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]

    target = raw_text[i]
    data.append((context, target))

print("data[:5] = {}\n".format(data[:5]))


# Continuous Bag-of-Word
class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(in_features=context_size * embedding_dim,
                                 out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def make_context_vector(context, word_to_ix):
    """
    It is a function to change <Int> idx to tensor.
    :param context: context word
    :param word_to_ix: index of the word
    :return: tensor(idx)
    """
    idxs = [word_to_idx[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

# print(make_context_vector(data[0][0], word_to_idx))


losses = []
loss_function = nn.NLLLoss()
# context, 2 words to the left, 2 words to the right
model = CBOW(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, context_size=CONTEXT_SIZE * 2)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):

    total_loss = 0
    for context, target in data:
        context_ids = make_context_vector(context, word_to_idx)
        model.zero_grad()

        log_probs = model(context_ids)
        label = torch.tensor([word_to_idx[target]], dtype=torch.long)

        loss = loss_function(log_probs, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    losses.append(total_loss)

print(losses)
print("\n")

print(make_context_vector(data[0][0], word_to_idx))
print(model.embeddings(make_context_vector(data[0][0], word_to_idx)))

print(model.embeddings(make_context_vector(data[0][0], word_to_idx)).size())
# torch.Size([4, 100])
