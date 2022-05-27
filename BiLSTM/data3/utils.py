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
    y_pred = torch.tensor(y_pred)
    # _, y_pred = torch.max(y_pred, dim=1)
    # y, y_pred = y.cpu(), y_pred.cpu()
    # non_pad_elements = (y != 0).nonzero().reshape(-1)  # prepare masking for paddings
    # correct = y_pred[non_pad_elements].eq(y[non_pad_elements])
    # P = correct.sum()/y_pred[(y_pred != 0).nonzero()].shape[0]
    # R = correct.sum() / y[non_pad_elements].shape[0]
    P = y_pred.eq(y).sum().item() / len(y_pred)
    print(y_pred.eq(y))
    R = y_pred.eq(y).sum().item() / len(y)
    f1_score = 2 * P * R / (P + R)
    if not test:
        print('Precision: {:.4f} | Recall: {:.4f} | f1_score: {:.4f}'.format(P, R, f1_score))
    return P, R, f1_score

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
        score = -val_loss

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

def cal_weight(y1):
    empty_list = []
    for i in range(len(y1)):
        empty_list+=y1[i]
    empty_list = np.array(empty_list)
    weight_numpy = np.ones(8)
    weight_numpy[0] = 0
    for i in range(1, 8):
        weight_numpy[i] = 1 / (np.count_nonzero(empty_list == i)+0.01)
    return weight_numpy

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
    # _, y_pred = torch.max(y_pred, dim=1)
    # x_ = ['O', 'I-LOC', 'B-ORG', 'I-PER', 'B-LOC', 'B-PER', 'I-ORG']
    # y_ = ['O', 'I-LOC', 'B-ORG', 'I-PER', 'B-LOC', 'B-PER', 'I-ORG']
    # non_pad_elements = (y != 0).nonzero().reshape(-1)
    # y, y_pred = y[non_pad_elements].cpu().numpy(), y_pred[non_pad_elements].cpu().numpy()

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