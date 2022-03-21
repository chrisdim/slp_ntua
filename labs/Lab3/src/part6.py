#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:26:16 2021

@author: chrisdim
"""

import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

from sklearn import metrics
import json

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.100d.txt")
#EMBEDDINGS = os.path.join(EMB_PATH, "glove.twitter.27B.100d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 100

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 11
DATASET = "MR"  # options: "MR", "Semeval2017A"
#DATASET="Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
le = LabelEncoder()
y_train = le.fit_transform(y_train)  # EX1
y_test = le.fit_transform(y_test)  # EX1
n_classes = le.classes_.size  # EX1 - LabelEncoder.classes_.size

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X_train)
print(X_tfidf.shape)


# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)


# EX4 - Define our PyTorch-based DataLoader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)  # EX7
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)  # EX7

