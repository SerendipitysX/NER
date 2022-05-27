import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import jieba
import jieba.posseg as psg


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["Pos"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
        print("{} sentences in the dataset".format(len(self.sentences)))

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# 2 提取特征
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'word.lower()' : word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        #'postag': postag,
    }

    if i > 0:   # 前一个词
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            #'-1:postag': postag1,
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:    # 后一个词
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            #'+1:postag': postag1,
        })
    else:
        features['EOS'] = True
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def get_seg_features(sent):
    seg_feature = []

    for word in jieba.cut(sent):
        if len(word) == 1:
            seg_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    return seg_feature


def get_pos_features(sent):
    pos_feature = []
    for w in psg.cut(sent):
        if len(w.word) == 1:
            pos_feature.append(w.flag)
        else:
            tmp = [w.flag] * len(w.word)
            pos_feature.extend(tmp)
    return pos_feature


def read_from_file(file_path: str):
    words = []
    chunks = []
    poss = []
    tags = []
    sentences_label = ["Sentence: 0"]
    sent_id = 1

    for line in open(file_path):
        if line in ['\n', '\r\n']:
            # 在https://www.aclweb.org/anthology/W03-0419.pdf官方统计中，将-DOCSTART- -X- -X- O
            # 文档开始符号也看成了一个句子, 这里我们不把这个文档符号看成一个句子
            if len(sentences_label) != 1:
                sentences_label.pop()
                sentences_label.append("Sentence: {}".format(sent_id))
            sent_id += 1
        elif line.startswith("-DOCSTART-"):
            # 将文件开头的-DOCSTART- -X- -X- O不计入句子数目
            sent_id = sent_id - 1 if len(sentences_label) != 1 else sent_id
        else:
            word, pos, chunk, tag = line.strip().split()
            assert (len(line.split()) == 4)
            words.append(word)
            poss.append(pos)
            chunks.append(chunk)
            tags.append(tag)
            sentences_label.append(None)
    sentences_label.pop()
    assert (len(sentences_label) == len(tags))

    dataset = {"Sentence #": sentences_label,
               "Word": words,
               "Pos": poss,
               "Chunk": chunks,
               "Tag": tags}
    data = pd.DataFrame(dataset, columns=['Sentence #', 'Word', 'Pos', 'Chunk', 'Tag'])
    data = data.fillna(method="ffill")  # 方便后续group
    print("file {}, tokens {}".format(file_path, len(data['Word'])))
    return data


def bulid_dataset(model_name, ner_dataset_dir, char2idx=None, tag2idx=None, max_len=-50):

    """
    构建数据
    :param data:
    :return:
    """
    # dataset_dir="../data/dataset.pkl"
    data = read_from_file(ner_dataset_dir)
    getter = SentenceGetter(data)
    sentences = getter.sentences

    # print(sentences[:300])

    # 输入长度等长，统一设置为50
    if not char2idx:
        chars = list(set(data["Char"].values)) + ["#OOV#"] + ["PADDING"]
        char2idx = {w: i for i, w in enumerate(chars)}
        with open("../datasets/char2idx-{}".format(model_name), 'wb') as out_data:
            pickle.dump(char2idx, out_data)
    if not tag2idx:
        tags = list(set(data["Tag"].values)) + ["PADDING"]
        tag2idx = {t: i for i, t in enumerate(tags)}
        with open("../datasets/tag2idx-{}".format(model_name), 'wb') as out_data:
            pickle.dump(tag2idx, out_data)

    X = []
    for s in sentences:
        temp = []
        for w in s:
            if w[0] in char2idx.keys():
                temp.append(char2idx[w[0]])
            else:
                temp.append(char2idx["#OOV#"])
        X.append(temp)
    y = [[tag2idx[w[1]] for w in s] for s in sentences]

    # 填充标签
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=char2idx["PADDING"])
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["PADDING"])

    # 将label转为categorial
    y = [to_categorical(i, num_classes=len(tag2idx.keys())) for i in y]

    # idx2str
    idx2tag = {t: i for i, t in tag2idx.items()}
    idx2char = {w: i for i, w in char2idx.items()}

    n_word = len(idx2char.keys())
    n_tags = len(idx2tag.keys())

    return n_word, n_tags, max_len, idx2tag, idx2char, X, y, tag2idx, char2idx



def bulid_dataset_jieba(model_name, ner_dataset_dir, char2idx=None, tag2idx=None, pos2idx=None, max_len=-50):

    """
    构建数据
    :param data:
    :return:
    """
    # dataset_dir="../data/dataset.pkl"
    data = read_from_file(ner_dataset_dir)
    getter = SentenceGetter(data)
    sentences = getter.sentences

    X_seg = []
    pos = []
    for s in sentences:
        sent = [w[0] for w in s]
        sent_seg = get_seg_features("".join(sent))
        sent_pos = get_pos_features("".join(sent))
        X_seg.append(sent_seg)
        pos.append(sent_pos)

    # 输入长度等长，统一设置为50
    if not char2idx:
        chars = list(set(data["Char"].values)) + ["#OOV#"] + ["PADDING"]
        char2idx = {w: i for i, w in enumerate(chars)}
        with open("../datasets/char2idx-{}".format(model_name), 'wb') as out_data:
            pickle.dump(char2idx, out_data)
    if not tag2idx:
        tags = list(set(data["Tag"].values)) + ["PADDING"]
        tag2idx = {t: i for i, t in enumerate(tags)}
        with open("../datasets/tag2idx-{}".format(model_name), 'wb') as out_data:
            pickle.dump(tag2idx, out_data)
    if not pos2idx:
        all_pos = []
        for j in pos:
            all_pos.extend(j)
        all_pos = list(set(all_pos)) + ["#OOV#"] + ["PADDING"]
        pos2idx = {p: i for i, p in enumerate(all_pos)}
        with open("../datasets/pos2idx-{}".format(model_name), 'wb') as out_data:
            pickle.dump(pos2idx, out_data)

    #X_pos = [[pos2idx[i] for i in j] for j in pos]

    X = []
    for s in sentences:
        temp = []
        for w in s:
            if w[0] in char2idx.keys():
                temp.append(char2idx[w[0]])
            else:
                temp.append(char2idx["#OOV#"])
        X.append(temp)

    X_pos = []
    for j in pos:
        temp = []
        for i in j:
            if i in pos2idx.keys():
                temp.append(pos2idx[i])
            else:
                temp.append(pos2idx["#OOV#"])
        X_pos.append(temp)

    y = [[tag2idx[w[1]] for w in s] for s in sentences]

    # 填充标签
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=char2idx["PADDING"])
    X_seg = pad_sequences(maxlen=max_len, sequences=X_seg, padding="post", value=4)
    X_pos = pad_sequences(maxlen=max_len, sequences=X_pos, padding="post", value=pos2idx["PADDING"])
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["PADDING"])
    # print("".join([s[0] for s in sentences[-1]]))
    # print(X[-1])
    # print(X_seg[-1])
    # print(X_pos[-1])
    # print(y[-1])
    # print(tag2idx)
    # print(pos2idx)

    # 将label转为categorial
    y = [to_categorical(i, num_classes=len(tag2idx.keys())) for i in y]

    # idx2str
    idx2tag = {t: i for i, t in tag2idx.items()}
    idx2char = {w: i for i, w in char2idx.items()}

    n_word = len(idx2char.keys())
    n_tags = len(idx2tag.keys())
    n_pos = len(pos2idx.keys())

    return n_word, n_pos, n_tags, max_len, idx2tag, idx2char, [X, X_pos, X_seg], y, tag2idx, char2idx, pos2idx

if __name__ == '__main__':
    # train_data = read_from_file("../data/train.txt")
    # valid_data = read_from_file("../data/valid.txt")
    # test_data = read_from_file("../data/test.txt")
    #
    #
    # getter = SentenceGetter(train_data)
    # sentences = getter.sentences
    # getter = SentenceGetter(valid_data)
    # sentences = getter.sentences
    # getter = SentenceGetter(test_data)
    # sentences = getter.sentences

    train_dir = '../data/train.txt'
    test_dir = '../data/test.txt'
    n_words, n_tags, n_pos, max_len, idx2tag, idx2word, X, X_pos, y, tags2idx, word2idx = bulid_dataset(train_dir, "../data/train.pkl", max_len=130)


    n_words, n_tags, n_pos, _, idx2tag, idx2word, X_test, X_pos_test, y_test, _, _ = bulid_dataset(test_dir, "../data/test.pkl", word2idx=word2idx, tag2idx=tags2idx, max_len=130)



