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
from gensim.models import KeyedVectors #for Google

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__)) + "/GoogleNews-vectors-negative300.bin"
google_model = KeyedVectors.load_word2vec_format(SCRIPT_DIRECTORY, binary=True,
limit=100000)

v1 = google_model.most_similar(positive=["bible"], topn=3)
v2 = google_model.most_similar(positive=["book"], topn=3)
v3 = google_model.most_similar(positive=["bank"], topn=3)
v4 = google_model.most_similar(positive=["water"], topn=3)

c1 = google_model.most_similar(positive=['girls', 'kings'], negative=['queen'])
c2 = google_model.most_similar(positive=['taller', 'good'], negative=['tall'])
c3 = google_model.most_similar(positive=['France', 'London'], negative=['Paris'])
