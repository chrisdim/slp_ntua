#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Speech & Natural Language Processing
# Laboratory 1 - OpenFST Spell Checker & Familiarization with Word2vec
# Autumn 2020 - 7th Semester

# Dimopoulos Christos [031 17 037] - chrisdim99@gmail.com
# Dimos Dimitris      [031 17 165] - dimitris.dimos647@gmail.com

import re
import sys
import contractions
import nltk
from string import ascii_lowercase
import os

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))   # points at lab1/scripts
main_directory = CURRENT_DIRECTORY


# PART 1 - Spell Checker Construction



# Step 1: Corpus Construction

def download_corpus(corpus="gutenberg"):
    """Download Project Gutenberg corpus, consisting of 18 classic books
    Book list:
       ['austen-emma.txt',
        'austen-persuasion.txt',
        'austen-sense.txt',
        'bible-kjv.txt',
        'blake-poems.txt',
        'bryant-stories.txt',
        'burgess-busterbrown.txt',
        'carroll-alice.txt',
        'chesterton-ball.txt',
        'chesterton-brown.txt',
        'chesterton-thursday.txt',
        'edgeworth-parents.txt',
        'melville-moby_dick.txt',
        'milton-paradise.txt',
        'shakespeare-caesar.txt',
        'shakespeare-hamlet.txt',
        'shakespeare-macbeth.txt',
        'whitman-leaves.txt']
    """
    nltk.download(corpus)
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
    lines = [preprocess(ln) for ln in corpus.split("\n")]
    lines = [ln for ln in lines if len(ln) > 0]  # Ignore empty lines

    return lines

if __name__ == "__main__":
    CORPUS = sys.argv[1] if len(sys.argv) > 1 else "gutenberg"
    raw_corpus = download_corpus(corpus=CORPUS)
    preprocessed = process_file(raw_corpus, preprocess=preprocess)
   
   
   
# Step 2: Dictionary Construction (new code from now on)
    
    # a) create dictionary
    dictionary = dict()  
    for line in preprocessed:
        for token in line:
            if token not in dictionary:
                dictionary[token] = 1
            else:
                dictionary[token] += 1
    
    # b) remove tokens that appared less than 5 times
    dictionary = { token:freq for token,freq in dictionary.items() if freq >= 5 }
    
    # c) create words.vocab.txt file
    words = open(main_directory+"/../vocab/words.vocab.txt", "w")
    dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
    for token in dictionary.keys():
        words.write(token)
        words.write("\t")
        words.write(str(dictionary[token]))
        words.write("\n")
    words.close()
    
    
    
# Step 3: Creating I/O symbols

    # a) creating chars.syms file
    index = 1
    chars = open(main_directory+"/../vocab/chars.syms", "w")
    chars.write("<eps>")
    chars.write("\t")
    chars.write(str(0))
    chars.write("\n")
    for letter in ascii_lowercase:
        chars.write(letter)
        chars.write("\t")
        chars.write(str(index))
        chars.write("\n")
        index += 1
    chars.close()
    
    # b) creating words.syms file
    index = 1
    words = open(main_directory+"/../vocab/words.syms", "w")
    words.write("<eps>")
    words.write("\t")
    words.write(str(0))
    words.write("\n")
    for token in dictionary.keys():
        words.write(token)
        words.write("\t")
        words.write(str(index))
        words.write("\n")
        index += 1
    words.close()
    
    

# Step 4: Creating a distance edit transducer

levenshtein = open(main_directory+"/../fsts/L.fst", "w")
# char to itself with weight 0 (no edit)
for letter in ascii_lowercase:
    levenshtein.write("0 0 "+letter+" "+letter+" 0\n")
# char to <eps> with weight 1 (deletion)
for letter in ascii_lowercase:
    levenshtein.write("0 0 "+letter+" "+"<eps>"+" 1\n")
# <eps> to char with weight 1 (insertion)
for letter in ascii_lowercase:
    levenshtein.write("0 0 "+"<eps>"+" "+letter+" 1\n")
# char to other chars with weight 1
for letter1 in ascii_lowercase:
    for letter2 in ascii_lowercase:
        if letter1 != letter2: 
            levenshtein.write("0 0 "+letter1+" "+letter2+" 1\n")
levenshtein.write(str(0))
levenshtein.close()



# Step 5: Creating a dictionary acceptor

def create_acceptor(dictionary, file):  # converts a dictionary into an FST acceptor                                      
    index = 1                           # and copies the acceptor to a file
    for word in dictionary.keys():
        split = [char for char in word]
        file.write("0 "+str(index)+" <eps> <eps> 0\n")
        file.write(str(index)+" "+str(index+1)+" "+split[0]+" "+word+" 0\n")
        for char in split[1:]:
            index += 1
            file.write(str(index)+" "+str(index+1)+" "+char+" <eps>"+" 0\n")
        file.write(str(index+1)+"\n")
        index += 2   
    return
    
"""                            
acceptor = open(main_directory+"test_acceptor.fst", "w")
test_dictionary = {"emma" : 1, "by" : 1, "jane" : 1, "volume" : 1, "i" : 1}
create_acceptor(test_dictionary, acceptor)
"""    

acceptor = open(main_directory+"/../fsts/V.fst", "w")
create_acceptor(dictionary, acceptor)
acceptor.close()

    