#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Speech & Natural Language Processing
# Laboratory 1 - OpenFST Spell Checker & Familiarization with Word2vec
# Autumn 2020 - 7th Semester

# Dimopoulos Christos [031 17 037] - chrisdim99@gmail.com
# Dimos Dimitris      [031 17 165] - dimitris.dimos647@gmail.com

import glob
import os
import re
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import KeyedVectors #for Google
import numpy as np
import sklearn


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__)) 
data_dir = os.path.join(SCRIPT_DIRECTORY, "../data/aclImdb/")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
pos_train_dir = os.path.join(train_dir, "pos")
neg_train_dir = os.path.join(train_dir, "neg")
pos_test_dir = os.path.join(test_dir, "pos")
neg_test_dir = os.path.join(test_dir, "neg")

# For memory limitations. These parameters fit in 8GB of RAM.
# If you have 16G of RAM you can experiment with the full dataset / W2V
MAX_NUM_SAMPLES = 5000
# Load first 1M word embeddings. This works because GoogleNews are roughly
# sorted from most frequent to least frequent.
# It may yield much worse results for other embeddings corpora
NUM_W2V_TO_LOAD = 1000000


SEED = 42

# Fix numpy random seed for reproducibility
np.random.seed(SEED)

class W2VLossLogger(CallbackAny2Vec):
    """Callback to print loss after each epoch
    use by passing model.train(..., callbacks=[W2VLossLogger()])
    """

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()

        if self.epoch == 0:
            print("Loss after epoch {}: {}".format(self.epoch, loss))
        else:
            print(
                "Loss after epoch {}: {}".format(
                    self.epoch, loss - self.loss_previous_step
                )
            )
        self.epoch += 1
        self.loss_previous_step = loss


def strip_punctuation(s):
    return re.sub(r"[^a-zA-Z\s]", " ", s)


def preprocess(s):
    return re.sub("\s+", " ", strip_punctuation(s).lower())


def tokenize(s):
    return s.split(" ")


def preproc_tok(s):
    return tokenize(preprocess(s))


def read_samples(folder, preprocess=lambda x: x):
    samples = glob.iglob(os.path.join(folder, "*.txt"))
    data = []

    for i, sample in enumerate(samples):
        if MAX_NUM_SAMPLES > 0 and i == MAX_NUM_SAMPLES:
            break
        with open(sample, "r") as fd:
            x = [preproc_tok(l) for l in fd][0]
            data.append(x)

    return data


def create_corpus(pos, neg):
    corpus = np.array(pos + neg)
    y = np.array([1 for _ in pos] + [0 for _ in neg])
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)

    return list(corpus[indices]), list(y[indices])


def extract_nbow(corpus, model):
    """Extract neural bag of words representations"""
    vecsize = np.size(model.wv[(corpus[0][0])])
    vocab_vectors = model.wv
    out = []
    for token in corpus:
        sentence_vec = np.zeros(vecsize)
        count = np.size(token)
        for word in token:
            if (word not in vocab_vectors): #OOV words
                nbow = np.zeros(vecsize)
            else:
                nbow = model.wv[word]
            sentence_vec += nbow
        out.append(sentence_vec/count)
    return out


def train_sentiment_analysis(train_corpus, train_labels):
    """Train a sentiment analysis classifier using NBOW + Logistic regression"""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(np.array(train_corpus), np.array(train_labels))
    return clf

def evaluate_sentiment_analysis(classifier, test_corpus, test_labels):
    """Evaluate classifier in the test corpus and report accuracy"""
    from sklearn.metrics import accuracy_score
    Xtest = StandardScaler().fit_transform(np.array(test_corpus)) # NORMALIZE
    return accuracy_score(np.array(test_labels), classifier.predict(Xtest))


if __name__ == "__main__":
    # TODO: read Imdb corpus
    positive = read_samples(pos_train_dir, preprocess)
    negative = read_samples(neg_train_dir, preprocess)
    train_corpus, train_labels = create_corpus(positive,negative)
    
    positive = read_samples(pos_test_dir, preprocess)
    negative = read_samples(neg_test_dir, preprocess)
    test_corpus, test_labels = create_corpus(positive,negative)
    
    # Load embeddings model we created
    output_file = "gutenberg_w2v.100d.model"
    model = Word2Vec.load(output_file)
    
    """
    #GOOGLE MODEL:
    SCRIPT_DIRECTORY = SCRIPT_DIRECTORY +"/GoogleNews-vectors-negative300.bin"
    model = KeyedVectors.load_word2vec_format(SCRIPT_DIRECTORY, binary=True,
    limit=100000)
    """
    
    # Create NBOW
    nbow_train_corpus = extract_nbow(train_corpus, model)
    nbow_test_corpus = extract_nbow(test_corpus, model)


    # TODO: train / evaluate and report accuracy
    
    # Train Linear Regression Classifier
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(np.array(nbow_train_corpus)) # NORMALIZE
    clf = train_sentiment_analysis(X_scaled, np.array(train_labels))


    # Calculate & Print Acurracy of Classifier we just trained
    scaled = StandardScaler().fit_transform(np.array(nbow_test_corpus))
    accuracy = evaluate_sentiment_analysis(clf, scaled, test_labels)
    print("==================================\nAccuracy_score: {}\n==================================\n"
      .format(accuracy))
