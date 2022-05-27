import torch
from TorchCRF import CRF
import TorchCRF
# from utils import *
from itertools import zip_longest
import torch.nn as nn
START_TAG, END_TAG = "<s>", "<e>"

# device = torch.device('cuda')


class LSTM_CRF(nn.Module):  # 注意Module首字母需要大写
    def __init__(self, vocab2id):
        super().__init__()
        self.embedding_size = 128
        self.input_size = self.embedding_size
        self.hidden_size = self.embedding_size
        self.tags_num = 7  # 根据 label2id 来的  {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6}
        self.embedding = nn.Embedding(len(vocab2id)+1, self.embedding_size, padding_idx=1)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=1,
                            bidirectional=True)
        self.hidden_tag_linear = nn.Linear(2 * self.hidden_size, self.tags_num)
        nn.init.kaiming_normal_(self.hidden_tag_linear.weight, mode='fan_in',
                                nonlinearity='relu')
        # nn.init.kaiming_normal_(self.linear.weight, mode='fan_in',
        #                         nonlinearity='relu')

        self.crf = CRF(self.tags_num)

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    def forward(self, x, y, mask):
        x = self.embedding(x) # 128,512,8
        x = x.transpose(0, 1) # 512,128,8

        lstm_out, (h_n, h_c) = self.lstm(x, None)
        logits = lstm_out.transpose(0, 1)

        logits = self.hidden_tag_linear(logits)

        crf_loss = self.crf(logits, y, mask)

        targets = self.crf.viterbi_decode(logits, mask=mask)
        return targets, crf_loss, logits