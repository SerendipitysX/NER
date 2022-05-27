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
    print('Using CPU')
### CN
id2label,label2id = get_string2id('dataset/CN/labels.txt')
id2vocab,vocab2id = get_string2id('dataset/CN/vocab.txt')
test_text,test_label = read_data('dataset/CN/example.test', vocab2id, label2id, max_len)

### EN
# id2label,label2id = get_string2id('dataset/EN/labels.txt')
# id2vocab,vocab2id = get_string2id('dataset/EN/vocab.txt')
# test_text,test_label = read_data('dataset/EN/test1.txt', vocab2id, label2id, max_len)

data = mydataset(test_text,test_label)
test_loader = DataLoader(data, batch_size=128, shuffle=True, num_workers=0, drop_last=True)

# model_ckpt = torch.load('/home/xss/NER/BiLSTM/data3/checkpoint/bilstm-crf/CN-best_model_f90.pth').cpu()#.state_dict()
# # print(model_ckpt["model_state"])
# print(type(model_ckpt))
# for key, value in state_dict.items():
#     print(key)

model = LSTM_CRF(vocab2id).to(device)
model.load_state_dict(torch.load('/home/xss/NER/BiLSTM/data3/checkpoint/focal-loss/CN-A-best_model.pth', map_location=torch.device('cpu')).cpu().state_dict())
model.eval()
label2id = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
test_epoch_loss = []
test_epoch_accuracy = []
with torch.no_grad():
    total_y = []
    total_pred = []
    for step, (train_x, train_y) in enumerate(test_loader):
        train_y = train_y.to(device)
        train_x = train_x.to(device)
        mask = torch.logical_not(torch.eq(train_x, torch.tensor(len(id2vocab))))
        targets_pred_without_pad, crf_loss,_ = model(train_x, train_y, mask)
        crf_loss = -crf_loss.mean()
        batch_accuracy_score = 0
        for batch_idx, batch_pre_targets in enumerate(targets_pred_without_pad):
            targets_true_without_pad = torch.masked_select(train_y[batch_idx], mask[batch_idx]).cpu().numpy()
            x_without_pad = torch.masked_select(train_x[batch_idx], mask[batch_idx]).cpu().numpy()
            #### x
            # juzi = []
            # for i in x_without_pad:
            #     juzi.append(id2vocab[i])
            #### y
            for i in targets_true_without_pad:
                total_y.append(i)
            #### y_pred
            pre_label = []
            for i in batch_pre_targets:
                pre_label.append(label2id[i])
                total_pred.append(i)

            # print("真实句子", juzi)  # ''.join(juzi)
            # print("预测标签", pre_label)

            # metric = accuracy_score(targets_true_without_pad, batch_pre_targets)
            # print(metric)
            # print("-" * 100)
        # break

df1 = pd.DataFrame({'y':total_y,'y_pred':total_pred})
A1 = accuracy_score(df1['y'],df1['y_pred'])
A2 = len(df1.loc[(df1.y==df1.y_pred)])/len(df1)
P = len(df1.loc[(df1.y!=0) & (df1.y==df1.y_pred)])/len(df1.loc[df1.y_pred!=0])
R = len(df1.loc[(df1.y!=0) & (df1.y==df1.y_pred)])/len(df1.loc[df1.y!=0])
F1 = 2*P*R/(P+R)
print("准确率：",A1)
print("精确率：",P)
print("召回率：",R)
print("F1值：",F1)

confuse_matrix = confusion_matrix(df1['y'],df1['y_pred'])[1:,1:]
print(confuse_matrix)

df2 = pd.DataFrame(confuse_matrix, columns=[ 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'])
fig_ = sns.heatmap(df2, linewidths=2, linecolor='white',annot=True,cmap="vlag",alpha=0.8) #YlGnBu Blues  vlag
fig_.set_yticklabels(['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'])
fig_.set_xlabel("True Label", fontsize = 14)
fig_.set_ylabel("Predict Label", fontsize = 14)
plt.show()
fig = fig_.get_figure()
fig.savefig('cm.png', dpi=400)