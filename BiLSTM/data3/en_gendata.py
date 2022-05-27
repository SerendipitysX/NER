import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd


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
    print(data['Word'])
    data = data.fillna(method="ffill")  # 方便后续group
    print("file {}, tokens {}".format(file_path, len(data['Word'])))
    return data

class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),
                                                           s['Pos'].values.tolist(),
                                                           s['Tag'].values.tolist())]
        self.grouped = self.data.groupby('Sentence #').apply(agg_func)
        self.sentences = [s for s in self.grouped]
        print('{} sentences in the dataset'.format(len(self.sentences)))

    def get_next(self):
        try:
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
    }

    if i > 0:  # last one
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.istitle()': word1.istitle(),
            # '-1:word.isdigit()': word1.isdigit(),
            '-1:postag': postag1,
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word2 = sent[i+1][0]
        postag2 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word2.lower(),
            '+1:word.isupper()': word2.isupper(),
            '+1:word.istitle()': word2.istitle(),
            '+1:word.isdigit()': word2.isdigit(),
            '+:postag': postag2,
        })
    else:
        features['EOS'] = True
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


