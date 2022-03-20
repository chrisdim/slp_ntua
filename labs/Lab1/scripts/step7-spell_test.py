#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Speech & Natural Language Processing
# Laboratory 1 - OpenFST Spell Checker & Familiarization with Word2vec
# Autumn 2020 - 7th Semester

# Dimopoulos Christos [031 17 037] - chrisdim99@gmail.com
# Dimos Dimitris      [031 17 165] - dimitris.dimos647@gmail.com

        # Usage:
        #   step7-corrector_test.py SPELL_CHECKER_FST
        #   (SPELL_CHECKER is JUST THE NAME of the corrector
        #   make sure SPELL_CHECKER_FST is in the fsts directory)
        # Output:
        #   saves corrections in a file

import subprocess
import sys
import os

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))   # points at lab1/scripts
main_directory = CURRENT_DIRECTORY
spell_checker = sys.argv[1]



# Step 7: Spell Checker Test

spell_test = open(main_directory+"/../data/spell_test.txt", "r")
results = open(main_directory+"/../"+spell_checker+"_test_results.txt", "w")
correct = []
wrong   = []

# store the test input
for _ in range(0,20):   # take the first 20 test lines
    line = spell_test.readline()
    correct_word = line.split()[0]   
    correct_word = correct_word[:len(correct_word)-1]
    correct.append(correct_word)    # we keep the correct words in a list
    wrong.append(line.split()[1:])  # and the wrong ones in another
    
# now test and output results   
for i in range(0,20):
    results.write("Correct word: "+correct[i]+"\n")
    for word in wrong[i]:
        out = subprocess.Popen([main_directory+'/predict.sh', main_directory+'/../fsts/'+spell_checker, wrong[i][0]], 
               stdout=subprocess.PIPE, 
               stderr=subprocess.STDOUT)
        stdout,stderr = out.communicate()
        results.write(word+" -> ")
        results.write(stdout.decode('utf-8')+"\n")
spell_test.close()
