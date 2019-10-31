import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(1)

# Definitions
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

# Hyper-parameters
learning_rate = 0.001
num_epochs = 10

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# build a list of tuples.
# Each tuple is ([ word_i-2, word_i-1 ], target word)

trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2]) for i in range(len(test_sentence) - 2)]
print(trigrams[:3])

# set() will remove the duplicated elements
vocab = set(test_sentence)

# Word to index
word_to_idx = {word: i for i, word in enumerate(vocab)}
# print(word_to_idx)


# N-gram language model
class NGramLanguageModeler(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(in_features=context_size * embedding_dim, 
                                 out_features=128)
        self.linear2 = nn.Linear(in_features=128, 
                                 out_features=vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(vocab_size=len(vocab),
                             embedding_dim=EMBEDDING_DIM,
                             context_size=CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_loss = 0
    for context, target in trigrams:
        
        # Step1. Prepare the inputs to be passed to the model
        context_idxs = torch.tensor([word_to_idx[w] for w in context], dtype=torch.long)
        print("context_idx = {}\n".format(context_idxs))

        # Step2. Recall that torch *accumulates* gradients. Before passing in a new instance, zero the old instance.
        model.zero_grad()

        # Step3. Run the forward pass, getting log probabilities over next words
        log_probs = model(context_idxs)

        # Step4. Compute the loss function.
        loss = loss_function(log_probs, torch.tensor([word_to_idx[target]], dtype=torch.long))

        # Step5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    losses.append(total_loss / len(word_to_idx))


print("losses = ", losses)

