#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Speech & Natural Language Processing
# Laboratory 1 - OpenFST Spell Checker & Familiarization with Word2vec
# Autumn 2020 - 7th Semester

# Dimopoulos Christos [031 17 037] - chrisdim99@gmail.com
# Dimos Dimitris      [031 17 165] - dimitris.dimos647@gmail.com

from string import ascii_lowercase
import math
import os
import subprocess

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))   # points at lab1/scripts
main_directory = CURRENT_DIRECTORY


# Step 8: Edit cost evaluation

# d)
wiki  = open(CURRENT_DIRECTORY+"/../data/wiki.txt", "r")
edits = open(CURRENT_DIRECTORY+"/../edits_8d.txt", "w")

for line in wiki: 
    out = subprocess.Popen([CURRENT_DIRECTORY+'/word_edits.sh', line.split()[0], line.split()[1]], 
               stdout=subprocess.PIPE, 
               stderr=subprocess.STDOUT)
    stdout,stderr = out.communicate()
    edits.write(stdout.decode('utf-8'))
wiki.close()
edits.close()

# e)
edit_frequencies = dict()
edits = open(CURRENT_DIRECTORY+"/../edits_8d.txt", "r")

count = 0
for line in edits:
    count += 1
    before = line.split()[0]
    after = line.split()[1]
    if (before,after) not in edit_frequencies:
        edit_frequencies[(before,after)] = 1
    else:
        edit_frequencies[(before,after)] += 1
"""
# some characters are not in the chars.syms so we remove them from dictionary
del edit_frequencies[("FATAL:", "FstCompiler:")]
del edit_frequencies[("ERROR:", "FstHeader::Read:")]
"""

# replace every frequency with the negative log
save = edit_frequencies.copy()
edit_frequencies.update((x, '%.3f'%(-math.log(y/count))) for x, y in edit_frequencies.items())
edits.close()

# st)
weighted_levenshtein = open(main_directory+"/../fsts/E.fst", "w")

for letter in ascii_lowercase:
    weighted_levenshtein.write("0 0 "+letter+" "+letter+" "+str(0)+"\n")

for letter in ascii_lowercase:
    if (letter,"<eps>") in edit_frequencies.keys():
        weighted_levenshtein.write("0 0 "+letter+" "+"<eps>"+" "+str(edit_frequencies[(letter,"<eps>")])+"\n")
    else:
        weighted_levenshtein.write("0 0 "+letter+" "+"<eps>"+" "+str(100000)+"\n")

for letter in ascii_lowercase:
    if ("<eps>", letter) in edit_frequencies.keys():
        weighted_levenshtein.write("0 0 "+"<eps>"+" "+letter+" "+str(edit_frequencies[("<eps>",letter)])+"\n")
    else:
        weighted_levenshtein.write("0 0 "+"<eps>"+" "+letter+" "+str(100000)+"\n")

for letter1 in ascii_lowercase:
    for letter2 in ascii_lowercase:
        if (letter1, letter2) in edit_frequencies.keys(): 
            weighted_levenshtein.write("0 0 "+letter1+" "+letter2+" "+str(edit_frequencies[(letter1,letter2)])+"\n")
        else:
            weighted_levenshtein.write("0 0 "+letter1+" "+letter2+" "+str(100000)+"\n")
            
weighted_levenshtein.write(str(0))
weighted_levenshtein.close()



# Step 9: Intro to word frequency (unigram word model)

sum = 0
vocab_file = open(main_directory+"/../vocab/words.vocab.txt", "r")
for line in vocab_file:
    sum += int(line.split()[1])
vocab_file.close()
vocab_file = open(main_directory+"/../vocab/words.vocab.txt", "r")
acceptor_W = open(main_directory+"/../fsts/W.fst", "w")
for line in vocab_file:
    acceptor_W.write("0 0 "+line.split()[0]+" "+line.split()[0]+" "+str('%.3f'%(-math.log((int(line.split()[1]))/sum)))+"\n")
acceptor_W.write("0")
vocab_file.close()
acceptor_W.close()

