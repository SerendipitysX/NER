import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def score(y_pred, y, test=False):
    _, y_pred = torch.max(y_pred, dim=1)
    y, y_pred = y.cpu(), y_pred.cpu()
    non_pad_elements = (y != 0).nonzero().reshape(-1)  # prepare masking for paddings
    correct = y_pred[non_pad_elements].eq(y[non_pad_elements])
    P = correct.sum()/y_pred[(y_pred != 0).nonzero()].shape[0]
    R = correct.sum() / y[non_pad_elements].shape[0]
    f1_score = 2 * P * R / (P + R)
    if not test:
        print('Precision: {:.4f} | Recall: {:.4f} | f1_score: {:.4f}'.format(P, R, f1_score))
    return P, R, f1_score


def read_from_file(file_path: str):
    words = []
    tags = []
    sentences_label = ["Sentence: 0"]
    sent_id = 1
    
    # 对于文件中的每一行，读取字，字标签，
    for line in open(file_path):
        if line.strip() == "":
            if len(sentences_label) != 1:
                sentences_label.pop()
                sentences_label.append("Sentence: {}".format(sent_id))
            sent_id += 1
        else:
            word, tag = line.strip().split()
            assert (len(line.split()) == 2)
            words.append(word)
            tags.append(tag)
            sentences_label.append(None)
    sentences_label.pop()
    assert (len(sentences_label) == len(tags))
    # 得到dataset字典，Sentence字段为每一个句子的id，方便SentenceGetter方法进行分句
    dataset = {"Sentence #": sentences_label,
               "Char": words,
               "Tag": tags}
    data = pd.DataFrame(dataset, columns=['Sentence #', 'Char', 'Tag'])
    data = data.fillna(method="ffill")  # 方便后续group
    print("file {}, chars {}".format(file_path, len(data['Char'])))
    return data

class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["Char"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
        self.max_len = max([len(s) for s in self.sentences])
        print("{} sentences in the dataset, max sentence length {}".format(len(self.sentences), self.max_len))

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# def tensorAndpadding(x, max_len=200):
#     x = torch.LongTensor(x)
#     m = nn.ConstantPad1d((0, max_len-x.shape[0]), 0)
#     return m(x)

def bulid_dataset(model_name, ner_dataset_dir, char2idx=None, tag2idx=None, max_len=-50):

    """
    构建数据
    :param data:
    :return:
    """
    data = read_from_file(ner_dataset_dir)
    getter = SentenceGetter(data)
    sentences = getter.sentences
    print(data.Tag.value_counts())

    # Taking all char and tag to id (From data)
    if not char2idx:
        chars = ["PADDING"] + list(set(data["Char"].values))
        char2idx = {w: i for i, w in enumerate(chars)}
        with open("/home/xss/NER/BiLSTM/data/tmp/char2idx-{}".format(model_name), 'wb') as out_data:
            pickle.dump(char2idx, out_data)
    if not tag2idx:
        tags = ["PADDING"] + list(set(data["Tag"].values))
        tag2idx = {t: i for i, t in enumerate(tags)}
        with open("/home/xss/NER/BiLSTM/data/tmp/tag2idx-{}".format(model_name), 'wb') as out_data:
            pickle.dump(tag2idx, out_data)
            
    # Taking char and tag to idx for every sentence
    X = []
    for s in sentences:
        temp = []
        for w in s:
            if w[0] in char2idx.keys():
                temp.append(char2idx[w[0]])
            # else:
            #     temp.append(char2idx["#OOV#"])
        X.append(temp)
    y = [[tag2idx[w[1]] for w in s] for s in sentences]

    # # padding for variant sequences
    # X = torch.stack(list(map(partial(tensorAndpadding, max_len = 200),X)))
    # y = torch.stack(list(map(partial(tensorAndpadding, max_len = 200),y)))
    # y = y.type(torch.int64)
    #
    # # one-hot encoding
    # scatter_dim = len(y.size()) # 2
    # y_tensor = y.view(*y.size(), -1) # [18,200,1]
    # zeros = torch.zeros(*y.size(), 8, dtype=y.dtype)
    # y = zeros.scatter(scatter_dim, y_tensor, 1).type(torch.float)

    # 为了后续方便从id转化为提及字，还需要生成提及字id到字映射字典以及字标签id到标签的映射字典：
    idx2tag = {t: i for i, t in tag2idx.items()}
    idx2char = {w: i for i, w in char2idx.items()}
    
    # n_word 字典中字数目
    n_word = len(idx2char.keys())
    # 标签数目
    n_tags = len(idx2tag.keys())

    return n_word, n_tags, max_len, idx2tag, idx2char, X, y, tag2idx, char2idx

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, path='/home/xss/NER/BiLSTM/data/checkpoint/bilstm.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 30
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
            path (str): Path for the checkpoint to be saved to. Default: 'checkpoint.pt'
            trace_func (function): trace print function. Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        # score = -val_loss
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



# def FocalLoss(y_pred, y):
    # # one-hot encoding
    # scatter_dim = len(y.size())  # 2
    # y_tensor = y.view(*y.size(), -1)  # [18,200,1]
    # zeros = torch.zeros(*y.size(), 8, dtype=y.dtype)
    # y = zeros.scatter(scatter_dim, y_tensor, 1).type(torch.float)
    # print(y.shape)
    # # calculate log
    # log_p = F.log_softmax(logits)
    # pt = label_onehot * log_p
    # sub_pt = 1 - pt
    # fl = -self.alpha * (sub_pt) ** self.gamma * log_p
    # if self.size_average:
    #     return fl.mean()
    # else:
    #     return fl.sum()

    # return loss

class FocalLoss(nn.Module):
    def __init__(self,weight_numpy):
        super(FocalLoss, self).__init__()
        self.gamma = 2
        self.alpha = torch.tensor(weight_numpy,dtype=torch.float).cuda()

    def forward(self, y_pred, y):
        non_pad_elements = (y != 0).nonzero().reshape(-1)
        y = y[non_pad_elements]
        y_pred = y_pred[non_pad_elements]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(y_pred, dim=-1)
        # print(log_p.is_cuda, y.is_cuda)
        ce = nn.NLLLoss(weight=self.alpha, reduction='none', ignore_index=0)(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(y_pred))
        log_pt = log_p[all_rows, y]

        pt = log_pt.exp()
        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = ((1 - pt) ** self.gamma * ce).mean()

        return loss


def confuse_matrix(y_pred, y):
    _, y_pred = torch.max(y_pred, dim=1)
    x_ = ['O', 'I-LOC', 'B-ORG', 'I-PER', 'B-LOC', 'B-PER', 'I-ORG']
    y_ = ['O', 'I-LOC', 'B-ORG', 'I-PER', 'B-LOC', 'B-PER', 'I-ORG']
    non_pad_elements = (y != 0).nonzero().reshape(-1)
    y, y_pred = y[non_pad_elements].cpu().numpy(), y_pred[non_pad_elements].cpu().numpy()

    # confusion_matrix = np.zeros((7, 7), dtype=np.int32)
    # for i in range(len(x_)):
    #     for j in range(len(y_)):
    #         # print(j)
    #         # print((y_pred == i + 1)[np.where(y == j + 1)[0]])
    #         confusion_matrix[i][j] = (y == i+1)[np.where(y_pred == j+1)[0]].sum()
    # # print(y_pred[y_pred!=0])
    # # print(set(y_pred))
    #
    # fig, ax = plt.subplots()
    # im = ax.imshow(confusion_matrix)
    #
    # # Show all ticks and label them with the respective list entries
    # ax.set_xticks(np.arange(len(x_)), labels=x_)
    # ax.set_yticks(np.arange(len(y_)), labels=y_)
    #
    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")
    #
    # # Loop over data dimensions and create text annotations.
    # for i in range(len(x_)):
    #     for j in range(len(y_)):
    #         text = ax.text(j, i, confusion_matrix[i, j],
    #                        ha="center", va="center", color="w")
    #
    # fig.tight_layout()
    # plt.show()

    cm = confusion_matrix(y, y_pred, labels=list(range(8)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(8)))
    disp.plot()
    plt.show()