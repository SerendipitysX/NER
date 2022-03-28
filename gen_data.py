import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext.data import Field, BucketIterator
from torchtext.datasets import SequenceTaggingDataset
from torch.autograd import Variable


class DatasetIterater(Dataset):
    def __init__(self, X, y):
        self.text = X
        self.label = y

    def __getitem__(self, item):
        return self.text[item], self.label[item]

    def __len__(self):
        return len(self.text)

def tensorAndpadding(x, max_len=200):
    x = torch.LongTensor(x)
    m = nn.ConstantPad1d((0, max_len-x.shape[0]), 0)
    return m(x)

def collate_order(batch):
    X,y = [],[]
    for i in range(len(batch)):
        X.append(batch[i][0])
        y.append(batch[i][1])
    seq_lengths = torch.LongTensor(list(map(len, X)))
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_lengths = torch.where(seq_lengths > 200, torch.tensor(200), seq_lengths)

    X = [X[i] for i in perm_idx]
    y = [y[i] for i in perm_idx]
    #padding for variant sequences
    X = torch.stack(list(map(partial(tensorAndpadding, max_len=200), X)))
    y = torch.stack(list(map(partial(tensorAndpadding, max_len=200), y)))
    y = y.type(torch.int64)

    # one-hot encoding
    # scatter_dim = len(y.size())  # 2
    # y_tensor = y.view(*y.size(), -1)  # [18,200,1]
    # zeros = torch.zeros(*y.size(), 8, dtype=y.dtype)
    # y = zeros.scatter(scatter_dim, y_tensor, 1).type(torch.float)
    return X,y,seq_lengths

class Corpus(object):

  def __init__(self, input_folder, min_word_freq, batch_size):
    # list all the fields
    self.word_field = Field(lower=True)
    self.tag_field = Field(unk_token=None)
    # create dataset using built-in parser from torchtext
    self.train_dataset, self.val_dataset, self.test_dataset = SequenceTaggingDataset.splits(
        path=input_folder,
        train="train.tsv",
        validation="val.tsv",
        test="test.tsv",
        fields=(("word", self.word_field), ("tag", self.tag_field))
    )
    # convert fields to vocabulary list
    self.word_field.build_vocab(self.train_dataset.word, min_freq=min_word_freq)
    self.tag_field.build_vocab(self.train_dataset.tag)
    # create iterator for batch input
    self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
        datasets=(self.train_dataset, self.val_dataset, self.test_dataset),
        batch_size=batch_size
    )
    # prepare padding index to be ignored during model training/evaluation
    self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
    self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]

corpus = Corpus(
    input_folder="/home/xss/NER/BiLSTM/data/dataset/input/",
    min_word_freq=1,  # any words occurring less than 3 times will be ignored from vocab
    batch_size=32
)
print(f"Train set: {len(corpus.train_dataset)} sentences")
print(f"Val set: {len(corpus.val_dataset)} sentences")
print(f"Test set: {len(corpus.test_dataset)} sentences")