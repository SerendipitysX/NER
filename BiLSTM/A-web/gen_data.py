import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
import torch.nn as nn
# from torchtext.data import Field, BucketIterator
# from torchtext.datasets import SequenceTaggingDataset
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
label2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6}

def get_string2id(path):
    string2id = {}
    id2string = []
    with open(path,'r',encoding = 'utf-8') as f:
        for line in f:
            string2id[line.strip()] = len(string2id)
            id2string.append(line.strip())
    return id2string,string2id


def get_sequence_len(path):
    sequence_len = list()
    tmp_len = 0
    with open(path,'r',encoding = 'utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                sequence_len.append(tmp_len)
                tmp_len = 0
            else:
                tmp_len += 1
    return np.array(sequence_len)

# train_sequence_len = get_sequence_len('dataset/CN/example.train')
# dev_sequence_len = get_sequence_len('dataset/CN/example.dev')
# test_sequence_len = get_sequence_len('dataset/CN/example.test')
# max_len = max(max(train_sequence_len),max(dev_sequence_len),max(test_sequence_len))
# print(len(train_sequence_len))
# print(len(dev_sequence_len))
# print(len(test_sequence_len))
# print(max_len)

def read_data(path,vocab2id,label2id,max_len,bert=None):
    data_x = list()
    data_y = list()
    tmp_text = list()
    tmp_label = list()
    chars = []
    label_ids = []
    if bert != '' and bert !=None:
        tokenizer = BertTokenizer.from_pretrained(bert)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) != 0:
                    char, label = line.strip().split()
                    chars.append(char)
                    label_ids.append(label2idx[label])
                else:
                    input_ids = [tokenizer.convert_tokens_to_ids(c) for c in chars]
                    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
                    label_ids = [label2idx['O']] + label_ids + [label2idx['O']]
                    if len(input_ids) > max_len:
                        input_ids = [input_ids[0]] + input_ids[:max_len - 2] + [input_ids[-1]]
                        label_ids = [label_ids[0]] + label_ids[:max_len - 2] + [label_ids[-1]]
                    assert len(input_ids) == len(label_ids)
                    input_ids += [0] * (max_len - len(input_ids))
                    label_ids += [0] * (max_len - len(label_ids))
                    data_x.append(input_ids)
                    data_y.append(label_ids)
                    chars = []
                    label_ids = []
    else:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    if len(tmp_text) >= max_len:
                        tmp_text = tmp_text[:max_len]
                        tmp_label = tmp_label[:max_len]
                    else:
                        # tmp_text += [len(vocab2id)] * (max_len - len(tmp_text))
                        tmp_text += [0] * (max_len - len(tmp_text))
                        tmp_label += [0] * (max_len - len(tmp_label))
                    data_x.append(tmp_text)
                    data_y.append(tmp_label)
                    tmp_text = list()
                    tmp_label = list()
                    # tmp_text_bert = list()
                else:
                    line = line.split(' ')
                    line = [elem for elem in line if elem.strip()]
                    try:
                        # tmp_text_bert.append(line[0])
                        tmp_text.append(vocab2id[line[0]])
                        tmp_label.append(label2id[line[1]])
                    except:
                        print(line)
    print(u'{} include sequences {}'.format(path,len(data_x)))
    # return np.array(data_x),np.array(data_y)
    return data_x,data_y


class mydataset(Dataset):
    def __init__(self, train_text, train_label):
        self._x = torch.LongTensor(train_text)
        self._y = torch.LongTensor(train_label)
        self._len = len(train_text)

    def __getitem__(self, item):
        return self._x[item], self._y[item]

    def __len__(self):  # 返回整个数据的长度
        return self._len

