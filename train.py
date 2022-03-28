import os
import random
import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import argparse
from gen_data import *
from utils import *
from model import BiLSTM,BiLSTM2

parser = argparse.ArgumentParser(description='BiLSTM')
parser.add_argument('--enable_cuda', type=bool, default='True', help='enable CUDA, default as True')
parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilize experiment results')
parser.add_argument('--bz', type=int, default=1, help='batchsize')
parser.add_argument('--embed_size', type=int, default=8)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--num_layer', type=int, default=4)
parser.add_argument('--max_len', type=int, default=200, help='the max length of sentence')
parser.add_argument('--epoches', type=int, default=150)
parser.add_argument('--step_size', type=int, default=10)
parser.add_argument('--patience', type=int, default=20, help='early stop,How long to wait after last time validation loss improved.')
parser.add_argument('--lr', type=float, default=1, help='learning rate')
parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'focal_loss'])
args = parser.parse_args()


# =============================== environment ================================
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
def set_env(seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False ## find fittest convolution
    torch.backends.cudnn.deterministic = True ## keep experiment result stable

# =============================== dataset ================================
MODEL = "bilstm"
train_dir = '/home/xss/NER/BiLSTM/data/dataset/example.train'
dev_dir = '/home/xss/NER/BiLSTM/data/dataset/example.dev'
test_dir = '/home/xss/NER/BiLSTM/data/dataset/example.test'
# train_dir = '/home/xss/NER/BiLSTM/data/dataset/input/train.tsv'
# dev_dir = '/home/xss/NER/BiLSTM/data/dataset/input/val.tsv'
# test_dir = '/home/xss/NER/BiLSTM/data/dataset/input/test.tsv'
n_words1, n_tags1, max_len, idx2tag1, idx2char1, X1, y1, tags2idx, char2idx = bulid_dataset(MODEL, train_dir, max_len=200)
n_words2, n_tags2, _, idx2tag2, idx2char3, X2, y2, _, _ = bulid_dataset(MODEL, dev_dir, char2idx=char2idx, tag2idx=tags2idx, max_len=200)
n_words3, n_tags3, _, idx2tag2, idx2char3, X3, y3, _, _ = bulid_dataset(MODEL, test_dir, char2idx=char2idx, tag2idx=tags2idx, max_len=200)

train_data = DatasetIterater(X1, y1)
train_loader = DataLoader(dataset=train_data, batch_size = args.bz, shuffle=True, drop_last=True, num_workers=1,collate_fn=collate_order)
val_data = DatasetIterater(X2, y2)
val_loader = DataLoader(dataset=val_data, batch_size = args.bz, shuffle=True, drop_last=True, num_workers=1,collate_fn=collate_order)
test_data = DatasetIterater(X3, y3)
test_loader = DataLoader(dataset=test_data, batch_size = args.bz, shuffle=False, drop_last=True, num_workers=1,collate_fn=collate_order)
print(tags2idx)
print(len(y1))
empty_list = []
for i in range(len(y1)):
    empty_list+=y1[i]
empty_list = np.array(empty_list)
weight_numpy = np.ones(8)
weight_numpy[0] = 0
for i in range(1, 8):
    weight_numpy[i] = 1 / np.count_nonzero(empty_list == i)

# =============================== model ==============================
# model = BiLSTM(vob_size=n_words1+1, embed_size=args.embed_size, hidden_size=args.hidden_size, num_layers=args.num_layer,batchsize=args.bz,max_len=args.max_len).to(device)
model = BiLSTM2(
    input_dim=n_words1+1,
    embedding_dim=300,
    hidden_dim=64,
    output_dim=8,
    lstm_layers=2,
    emb_dropout=0.5,
    lstm_dropout=0.1,
    fc_dropout=0.25,
    word_pad_idx=0
).cuda()
model.init_weights()
model.init_embeddings(0)
print(f"The model has {model.count_parameters():,} trainable parameters.")
print(model)

# ========================== optimizer & loss ========================
optimizer = optim.SGD(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

# =============================== accuracy ================================
def accuracy(self, preds, y):
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    non_pad_elements = (y != self.data.tag_pad_idx).nonzero()  # prepare masking for paddings
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])

# =============================== train ================================
def train(score, epochs, optimizer, scheduler, early_stopping, model, train_loader, val_iter, weigth_list):
    min_val_loss = np.inf
    for epoch in range(epochs):
        model.train()
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        for x, y, seq_lengths in tqdm.tqdm(train_loader):
            # step1
            # print(x.shape,y.shape,seq_lengths)
            x, y = x.to(device), y.to(device)
            # print(y)
            optimizer.zero_grad()
            # step2
            # y_pred = model(x,seq_lengths)  # [batch_size, num_nodes]
            y_pred = model(x) #[bz,sentence_len,out]
            # step3
            # print(y_pred.shape,y.shape)
            # flatten pred_tags to [sent len, batch size, output dim]
            y_pred = y_pred.view(-1, y_pred.shape[-1])
            # print(y_pred.shape)
            # flatten true_tags to [sent len * batch size]
            y = y.view(-1)
            if args.loss == 'cross_entropy':  #weight=torch.tensor([0,0,0,0,0,0,1,0],dtype=torch.float).to(device),
                loss_func = nn.CrossEntropyLoss(ignore_index=0,weight=torch.tensor(weight_numpy,dtype=torch.float).to(device),reduction='mean')
                loss = loss_func(y_pred, y)
                # print(loss.shape,loss[:10])
            if args.loss == 'focal_loss':
                loss_func = FocalLoss(weight_numpy)
                loss = loss_func(y_pred, y)
            # step4
            loss.backward()
            ## EXP1 loss not ignore padding
            l_sum += loss.item()
            n += 1
        score(y_pred, y)
        optimizer.step()
        scheduler.step()
        val_loss, p, r, f = val(model, val_iter, loss_func, score)
        if val_loss < min_val_loss:
            min_val_loss = val_loss

        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.6f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))
        early_stopping(f, model)

        if early_stopping.early_stop:
            print("Early stopping.")
            break
    print('\nTraining finished.\n')

def val(model, val_loader, Loss_func, score):
    model.eval()
    l_sum, n = 0.0, 0
    total_y_pred, total_y = [], []
    with torch.no_grad():
        for x, y, seq_lengths in val_loader:
            x, y = x.to(device), y.to(device)
            # y_pred = model(x,seq_lengths)
            y_pred = model(x)
            y_pred = y_pred.view(-1, y_pred.shape[-1])
            y = y.view(-1)                    
            # l = Loss_func(y_pred.reshape(-1,8), y.reshape(-1,8))
            l = Loss_func(y_pred, y)
            l_sum += l.item()
            n += 1
            total_y_pred.append(y_pred)
            total_y.append(y)
        total_y_pred = torch.cat(total_y_pred, dim=0)
        total_y = torch.cat(total_y, dim=0)
        confuse_matrix(total_y_pred, total_y)
        p, r, f = score(y_pred, y, test=False)
        return l_sum / n, p, r, f

def test(model_save_path, model, test_loader, score):
    model.load_state_dict(torch.load(model_save_path))
    n = 0
    total_P, total_R, total_F = 0, 0, 0
    total_y_pred, total_y = [], []
    with torch.no_grad():
        for x, y, seq_lengths in test_loader:
            model.eval()
            x, y = x.to(device), y.to(device)
            y_pred = F.softmax(model(x,dropout=False))
            y_pred = y_pred.view(-1, y_pred.shape[-1])
            y = y.view(-1)
            n += 1
            p, r, f = score(y_pred, y, test = True)
            # print(p,r,f)
            total_P += p
            total_R += r
            total_F += f
            total_y_pred.append(y_pred)
            total_y.append(y)
    total_y_pred = torch.cat(total_y_pred, dim=0)
    total_y = torch.cat(total_y, dim=0)
    confuse_matrix(total_y_pred, total_y)
    print('Precision: {:.4f} | Recall: {:.4f} | f1_score: {:.4f}'.format(total_P/n, total_R/n, total_F/n))


if __name__ == '__main__':
    set_env(args.seed)
    train(score, args.epoches, optimizer, scheduler, EarlyStopping(patience=args.patience), model, train_loader, val_loader,weight_numpy)
    # test('/home/xss/NER/BiLSTM/data/checkpoint/bilstm.pth', model, test_loader, score)
