import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
from utils import *
from sklearn_crfsuite import CRF
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report
import eli5

ner_dataset_dir = 'datasets/datasets/train.txt'
data = read_from_file(ner_dataset_dir)
test_dir = 'datasets/datasets/test.txt'
test_data = read_from_file(test_dir)

getter = SentenceGetter(data)
sentences = getter.sentences
X = [sent2features(s) for s in sentences]
Y = [sent2labels(s) for s in sentences]

test_getter = SentenceGetter(test_data)
test_sentences = test_getter.sentences
test_X = [sent2features(s) for s in test_sentences]
test_Y = [sent2labels(s) for s in test_sentences]

def train():
    crf = CRF(algorithm='lbfgs', c1=10, c2=0.1, max_iterations=100, all_possible_transitions=True)
    pred = cross_val_predict(estimator=crf, X=X, y=Y, cv=3)
    report = flat_classification_report(y_pred=pred, y_true=Y)
    crf.fit(X, Y)

    print(report)
    eli5.show_weights(crf, top=5, show=['transition_features'])
    eli5.show_weights(crf, top=10, feature_re='^word\.is',
                      horizontal_layout=False, show=['targets'])

    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(crf, 'models/crf.pkl')


def test():
    crf = joblib.load(filename='models/crf.pkl')
    pred = crf.predict(test_X)
    report = flat_classification_report(y_pred=pred, y_true=test_Y)
    print(report)
    f_out = 'output/crf.output.txt'
    with open(f_out, 'w') as f:
        for s, s_pred in zip(sentences, pred):
            for w, p in zip(s, s_pred):
                f.write('{}\t{}\t{}\n'.format(w[0], w[2], p))


train()
test()

