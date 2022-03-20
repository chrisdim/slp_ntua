#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Speech & Natural Language Processing
# Laboratory 1 - OpenFST Spell Checker & Familiarization with Word2vec
# Autumn 2020 - 7th Semester

# Dimopoulos Christos [031 17 037] - chrisdim99@gmail.com
# Dimos Dimitris      [031 17 165] - dimitris.dimos647@gmail.com

import logging
import multiprocessing
import os
import sys
import contractions
import nltk
import re

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

########################## READ GUTENBERG AGAIN ##############################################
def download_corpus(corpus="gutenberg"):
    """Download Project Gutenberg corpus, consisting of 18 classic books
    """
    raw = nltk.corpus.__getattr__(corpus).raw()
    return raw
def identity_preprocess(s):
    return s
def clean_text(s):
    s = s.strip()  # strip leading / trailing spaces
    s = s.lower()  # convert to lowercase
    s = contractions.fix(s)  # e.g. don't -> do not, you're -> you are
    s = re.sub("\s+", " ", s)  # strip multiple whitespace
    s = re.sub(r"[^a-z\s]", " ", s)  # keep only lowercase letters and spaces
    return s
def tokenize(s):
    tokenized = [w for w in s.split(" ") if len(w) > 0]  # Ignore empty string
    return tokenized
def preprocess(s):
    return tokenize(clean_text(s))
def process_file(corpus, preprocess=identity_preprocess):
    lines = [preprocess(ln) for ln in corpus.split(".")]
    lines = [ln for ln in lines if len(ln) > 0]  # Ignore empty lines
    return lines
########################################################################


# Enable gensim logging
logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


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


def train_w2v_model(
    sentences,
    output_file,
    window=5,
    embedding_dim=100,
    epochs=1000,
    min_word_count=10,
):
    """Train a word2vec model based on given sentences.
    Args:
        sentences list[list[str]]: List of sentences. Each element contains a list with the words
            in the current sentence
        output_file (str): Path to save the trained w2v model
        window (int): w2v context size
        embedding_dim (int): w2v vector dimension
        epochs (int): How many epochs should the training run
        min_word_count (int): Ignore words that appear less than min_word_count times
    """
    workers = multiprocessing.cpu_count()
    model = Word2Vec(sentences, size = embedding_dim, window= window, min_count=min_word_count, 
                     workers=workers, callbacks=[W2VLossLogger()])
    model.train(sentences, total_examples=1, epochs=1000)
    model.save(output_file)
    
    # TODO: Instantiate gensim.models.Word2Vec class
    
    # TODO: Build model vocabulary using sentences
    # TODO: Train word2vec model
    # model.train(..., callbacks=[W2VLossLogger()])
    # Save trained model
    # model.save(output_file)

    return model


if __name__ == "__main__":
    # read data/gutenberg.txt in the expected format
    CORPUS = sys.argv[1] if len(sys.argv) > 1 else "gutenberg"
    raw_corpus = download_corpus(corpus=CORPUS)
    
    sentences = process_file(raw_corpus, preprocess=preprocess)
    output_file = "gutenberg_w2v.100d.model"
    window = 5
    embedding_dim = 100
    epochs = 1000
    min_word_count = 10

    
    model = train_w2v_model(
        sentences,
        output_file,
        window=window,
        embedding_dim=embedding_dim,
        epochs=epochs,
        min_word_count=min_word_count,
    )
    
    

    #model = Word2Vec.load(output_file)

    # Check some values and similarities
    v1 = model.most_similar(positive=["bible"], topn=3)
    v2 = model.most_similar(positive=["book"], topn=3)
    v3 = model.most_similar(positive=["bank"], topn=3)
    v4 = model.most_similar(positive=["water"], topn=3)
    
    c1 = model.most_similar(positive=['girls', 'kings'], negative=['queen'])
    c2 = model.most_similar(positive=['taller', 'good'], negative=['tall'])
    c3 = model.most_similar(positive=['france', 'london'], negative=['paris'])
    # Create tsv files
    import csv
    with open('embeddings.tsv', 'w') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        words = model.wv.vocab.keys()
        for word in words:
            vector = model.wv.get_vector(word).tolist()
            row = vector
            writer.writerow(row)
            
    with open('metadata.tsv', 'w') as tsvfile:
        writer = csv.writer(tsvfile)
        words = model.wv.vocab.keys()
        for word in words:
            row = [word]
            writer.writerow(row)
