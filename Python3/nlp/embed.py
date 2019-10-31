import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(1)

word_to_idx = {"hello": 0, "world": 1}

# 2 words in vocab, 5 dimensional embeddings
embeds = nn.Embedding(2, 5)
print("embeds = {}".format(embeds))
print("embeds.weight = {}\n".format(embeds.weight))

lookup_tensor = torch.tensor([word_to_idx["hello"]], dtype=torch.long)
print("lookup_tensor = ", lookup_tensor)

hello_tensor = torch.tensor(word_to_idx["hello"])
print("hello_tensor = ", hello_tensor)
world_tensor = torch.tensor(word_to_idx["world"])
print("world_tensor = ", world_tensor)

hello_embed = embeds(lookup_tensor)
print("hello_embed = {}\n".format(hello_embed))

hello_tensor_embed = embeds(hello_tensor)
world_tensor_embed = embeds(world_tensor)
print("hello_tensor_embed = {}\nworld_tensor_embed = {}".format(hello_tensor_embed, world_tensor_embed))


