import torch as torch
import numpy as np
import matplotlib.pyplot as plt
from model import *
from gen_data import *
import seaborn as sns
import pandas as pd
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')

id2label,label2id = get_string2id('dataset/CN/labels.txt')
id2vocab,vocab2id = get_string2id('dataset/CN/vocab.txt')

data_x = []
test_sentence = '金小诚在成都大运会担任志愿者，因为他是红十字协会成员'
for i in test_sentence:
    data_x.append(vocab2id[i])
data_x += [len(vocab2id)] * (max_len - len(data_x))
data_x = torch.LongTensor(data_x).unsqueeze(0)
print(data_x.shape)
data_x = torch.cat([data_x]*128,dim=0)
print(data_x.shape)

model = LSTM_CRF(vocab2id).to(device)
model.load_state_dict(torch.load('/home/xss/NER/BiLSTM/data3/checkpoint/bilstm-crf/CN-best_model_f90.pth').cpu().state_dict())
model.eval()
with torch.no_grad():
    data_x = data_x.to(device)
    mask = torch.logical_not(torch.eq(data_x, torch.tensor(4465)))
    data_y = torch.zeros((128,577),dtype=torch.long).to(device)
    targets_pred_without_pad, crf_loss, _ = model(data_x, data_y, mask)
    tags = [id2label[i] for i in targets_pred_without_pad[0]]
    print("真实句子", test_sentence)  # ''.join(juzi)
    print("预测标签", tags)

# def extract(chars, tags):
PER = []
LOC = []
ORG = []
chars = list(test_sentence)
for i in range(len(tags)):
    if tags[i] in ['B-PER', 'I-PER']:
        PER.append(chars[i])
    if tags[i] in ['B-LOC', 'I-LOC']:
        LOC.append(chars[i])
    if tags[i] in ['B-ORG', 'I-ORG']:
        ORG.append(chars[i])
if len(PER) != 0:
    print("PER", ''.join(PER))
if len(LOC) != 0:
    print("LOC", ''.join(LOC))
if len(ORG) != 0:
    print("ORG", ''.join(ORG))