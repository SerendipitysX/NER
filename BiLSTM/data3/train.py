import os
import nni
import random
import tqdm
import torch.optim as optim
from tqdm import trange
import argparse
from gen_data import *
from en_gendata import *
from utils import *
from model import LSTM_CRF
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from loss_helper import FocalLoss
from transformers import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


parser = argparse.ArgumentParser(description='BiLSTM')
parser.add_argument('--enable_cuda', type=bool, default='True', help='enable CUDA, default as True')
parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilize experiment results')
parser.add_argument('--bz', type=int, default=128, help='batchsize')
parser.add_argument('--embed_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--num_layer', type=int, default=4)
parser.add_argument('--max_len', type=int, default=512, help='the max length of sentence')
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--step_size', type=int, default=10)
parser.add_argument('--patience', type=int, default=20, help='early stop,How long to wait after last time validation loss improved.')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--loss', type=str, default='bilstm+crf', choices=['cross_entropy', 'focal_loss', 'bilstm+crf'])
parser.add_argument('--language', type=str, default='CN', choices=['CN', 'EN'])
parser.add_argument('--focalloss', type=bool, default=False, help='enable focal loss')
parser.add_argument('--bert', type=str, default='', choices=['','bert-base-chinese', 'bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 'bert-large-cased'])
args = parser.parse_args()

param = {'alpha': 0.25, 'gamma': 2, 'w':0.25}
# param = nni.get_next_parameter()
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
    torch.backends.cudnn.deterministic = False ## keep experiment result stable

set_env(42)
# =============================== dataset ================================
max_len = args.max_len
if args.language == 'CN':
    id2label,label2id = get_string2id('dataset/CN/labels.txt')
    id2vocab,vocab2id = get_string2id('dataset/CN/vocab.txt')
    train_text,train_label = read_data('dataset/CN/example.train', vocab2id, label2id, max_len, args.bert)
    dev_text,dev_label = read_data('dataset/CN/example.dev', vocab2id, label2id, max_len, args.bert)
    test_text,test_label = read_data('dataset/CN/example.test', vocab2id, label2id, max_len, args.bert)

    # pickle.dump(train_text, open("/home/xss/NER/BiLSTM/data3/dataset/CN/pickle/train_text.p", "wb" ))
    # pickle.dump(train_label, open("/home/xss/NER/BiLSTM/data3/dataset/CN/pickle/train_label.p", "wb" ))
    # pickle.dump(test_text, open("/home/xss/NER/BiLSTM/data3/dataset/CN/pickle/test_text.p", "wb"))
    # pickle.dump(test_label, open("/home/xss/NER/BiLSTM/data3/dataset/CN/pickle/test_label .p", "wb"))
    # train_text = pickle.load(open('/home/xss/NER/BiLSTM/data3/dataset/CN/pickle/train_text.p', 'rb'))
    # train_label = pickle.load(open('/home/xss/NER/BiLSTM/data3/dataset/CN/pickle/train_label.p', 'rb'))
    data = mydataset(train_text,train_label)
    train_data, test_data = random_split(data, [round(0.8 * data._len), round(0.2 * data._len)])
    test_loader = DataLoader(test_data, batch_size=args.bz, shuffle=True, num_workers=0, drop_last=True)
    train_loader = DataLoader(train_data, batch_size=args.bz, shuffle=True, num_workers=0, drop_last=True)
if args.language == 'EN':
    id2label,label2id = get_string2id('dataset/EN/labels.txt')
    id2vocab,vocab2id = get_string2id('dataset/EN/vocab.txt')
    train_text, train_label = read_data('dataset/EN/train1.txt', vocab2id, label2id, max_len, args.bert)
    test_text, test_label = read_data('dataset/EN/test1.txt', vocab2id, label2id, max_len, args.bert)
    data = mydataset(train_text, train_label)
    train_data, test_data = random_split(data, [round(0.8 * data._len), round(0.2 * data._len)])
    test_loader = DataLoader(test_data, batch_size=args.bz, shuffle=True, num_workers=0, drop_last=True)
    train_loader = DataLoader(train_data, batch_size=args.bz, shuffle=True, num_workers=0,  drop_last=True)

# =============================== model ==============================
model = LSTM_CRF(vocab2id).to(device)
model.init_weights()
# model.load_state_dict(torch.load('/home/xss/NER/BiLSTM/data3/checkpoint/bilstm-crf/CN-best_model_f90.pth').state_dict())
# model.load_state_dict(torch.load('checkpoint/bilstm-crf/EN-C-best_model.pth').state_dict())
print(model)

# ========================== optimizer & loss ========================
# optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # 建立优化器实例
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

# =============================== accuracy ================================
def eval_test(model):
    test_epoch_loss = []
    total_y = []
    total_pred = []
    label2id = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    with torch.no_grad():
        optimizer.zero_grad()
        for step, (train_x, train_y) in enumerate(test_loader):
            train_y = train_y.to(device)
            train_x = train_x.to(device)
            # print(train_x)
            # mask = torch.logical_not(torch.eq(train_x, torch.tensor(len(id2vocab))))
            mask = torch.logical_not(torch.eq(train_x, torch.tensor(21128)))
            targets_pred_without_pad, crf_loss, logits = model(train_x, train_y, mask)
            crf_loss = -crf_loss.mean()
            if args.focalloss:
                fl = FocalLoss(gamma=param['gamma'], alpha=param['alpha'])
                loss = crf_loss + param['w'] * fl(logits.permute(0, 2, 1), train_y)
            else:
                loss = crf_loss
            for batch_idx, batch_pre_targets in enumerate(targets_pred_without_pad):
                targets_true_without_pad = torch.masked_select(train_y[batch_idx], mask[batch_idx]).cpu().numpy()
                if len(targets_true_without_pad) != len(batch_pre_targets):
                    continue

                ### y
                for i in targets_true_without_pad:
                    total_y.append(i)
                #### y_pred
                for j in batch_pre_targets:
                    total_pred.append(j)

        df1 = pd.DataFrame({'y': total_y, 'y_pred': total_pred})
        A1 = accuracy_score(df1['y'], df1['y_pred'])
        P = len(df1.loc[(df1.y != 0) & (df1.y == df1.y_pred)]) / len(df1.loc[df1.y_pred != 0])
        R = len(df1.loc[(df1.y != 0) & (df1.y == df1.y_pred)]) / len(df1.loc[df1.y != 0])
        F1 = 2 * P * R / (P + R)
        # print("准确率：", A1)
        # print("精确率：", P)
        # print("召回率：", R)
        # print("F1值：", F1)
        metric = {'acc': A1, 'precision': P, 'recall': R, 'F1': F1}
        test_epoch_loss.append(loss.item())

    return np.mean(test_epoch_loss), metric

# =============================== train ================================
sum_train_epoch_loss = []
sum_test_epoch_loss = []

best_test_loss = 1000000
for epoch in range(args.epochs):
    train_epoch_loss = []
    for step, (train_x, train_y) in enumerate(tqdm.tqdm(train_loader)):
        train_y = train_y.to(device)
        train_x = train_x.to(device)
        mask = torch.logical_not(torch.eq(train_x, torch.tensor(0))).to(device)
        # bert = BertModel.from_pretrained('bert-base-chinese').to(device)
        # train_x = bert(train_x, attention_mask=mask).last_hidden_state         #  .pooler_output
        # train_x = nn.Linear(768, 128)(train_x)
        targets_pred_without_pad, crf_loss, logits = model(train_x, train_y, mask)  # logits: [bz, seq_len, num_tags]
        crf_loss = -crf_loss.mean()
        if args.focalloss:
            fl = FocalLoss(gamma=param['gamma'], alpha=param['alpha'])
            fl = fl(logits.permute(0, 2, 1), train_y)
            loss = crf_loss + param['w']*fl
        else:
            loss = crf_loss
        loss.backward()
        optimizer.step()
        for batch_idx, batch_pred_targets in enumerate(targets_pred_without_pad):
            targets_true_without_pad = torch.masked_select(train_y[batch_idx], mask[batch_idx]).cpu().numpy()
        train_epoch_loss.append(loss.item())

    train_epoch_loss = np.mean(train_epoch_loss)
    test_epoch_loss, metrics = eval_test(model)
    writer.add_scalar("Loss/train", train_epoch_loss, epoch)
    writer.add_scalar("Loss/train", test_epoch_loss, epoch)
    # nni.report_intermediate_result(metrics)

    if test_epoch_loss < best_test_loss:
        best_test_loss = test_epoch_loss
        print("best_test_loss", best_test_loss)
        best_model = model
    sum_train_epoch_loss.append(train_epoch_loss)
    sum_test_epoch_loss.append(test_epoch_loss)

    print("epoch:" + str(epoch) + "  train_epoch_loss： " + str(train_epoch_loss) + "  test_epoch_loss: " + str(
        test_epoch_loss))
    print(metrics)
    # torch.save(model, '/home/xss/NER/BiLSTM/data3/checkpoint/bert/EN-'+str(epoch)+'-'+'.pth')
# nni.report_final_result(test_epoch_loss)
torch.save(best_model, '/home/xss/NER/BiLSTM/data3/checkpoint/bert/CH-'+'A-best_model.pth')