import torchtext
from torchtext.data import get_tokenizer
from torchtext import data, datasets
import torch

import io
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab

dataset_path = "../../data/aclImdb/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


########## get_tokenizer ##########
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer("I am a master student of UCAS.")
print(tokens)


########## Field ##########
TEXT = data.Field(tokenize='spacy')
LABELS = data.LabelField()
print("TEXT = {}".format(TEXT))
print("LABELS = {}".format(LABELS))


########## TabularDataset ##########
train, test = data.TabularDataset.splits(path=dataset_path,
                                         train="train.tsv",
                                         test="test.tsv",
                                         fields=[('labels', LABELS), ('text', TEXT)],
                                         format="tsv")
print("train = {}".format(train))
print("test = {}".format(test))


########## BucketIterator ##########
train_iter, test_iter = data.BucketIterator.splits((train, test), batch_sizes=(16, 256, 256), sort_key=lambda x: len(x.text), device=device)
print("train_iter = {}".format(train_iter))
print("test_iter = {}".format(test_iter))


######### Build vocab ###########
TEXT.build_vocab(train)
LABELS.build_vocab(train)

print(vars(train[0]))
# {'text': ['pos'], 'labels': ['Bromwell', 'High', 'is', 'a', 'cartoon', 'comedy.', 'It', 'ran', 'at', 'the', 'same', 'time', 'as', 'some', 'other', 'programs', 'about', 'school', 'life,', 'such', 'as', '"Teachers".', 'My', '35', 'years', 'in', 'the', 'teaching', 'profession', 'lead', 'me', 'to', 'believe', 'that', 'Bromwell', "High's", 'satire', 'is', 'much', 'closer', 'to', 'reality', 'than', 'is', '"Teachers".', 'The', 'scramble', 'to', 'survive', 'financially,', 'the', 'insightful', 'students', 'who', 'can', 'see', 'right', 'through', 'their', 'pathetic', "teachers'", 'pomp,', 'the', 'pettiness', 'of', 'the', 'whole', 'situation,', 'all', 'remind', 'me', 'of', 'the', 'schools', 'I', 'knew', 'and', 'their', 'students.', 'When', 'I', 'saw', 'the', 'episode', 'in', 'which', 'a', 'student', 'repeatedly', 'tried', 'to', 'burn', 'down', 'the', 'school,', 'I', 'immediately', 'recalled', '.........', 'at', '..........', 'High.', 'A', 'classic', 'line:', 'INSPECTOR:', "I'm", 'here', 'to', 'sack', 'one', 'of', 'your', 'teachers.', 'STUDENT:', 'Welcome', 'to', 'Bromwell', 'High.', 'I', 'expect', 'that', 'many', 'adults', 'of', 'my', 'age', 'think', 'that', 'Bromwell', 'High', 'is', 'far', 'fetched.', 'What', 'a', 'pity', 'that', 'it', "isn't!"]}


######## Batch ##########
print('Train:')
for batch in train_iter:
    print(batch)

