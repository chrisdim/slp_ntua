#!/usr/bin/env bash

# Calculate the edits needed to get from a misspelled word to a correct word

# Usage:
#   bash scripts/word_edits.sh tst test
# Output:
#   <eps> e

# Command line args
WRONG=${1}
CORRECT=${2}


# Constants.
CURRENT_DIRECTORY=$(dirname $0)

###
# Make sure to create these files
CHARSSYMS=${CURRENT_DIRECTORY}/phones.syms  # Phone symbol table
VANILLA_LEVENSHTEIN=${CURRENT_DIRECTORY}/L.fst  # Compile basic Levenshtein FST
###

# Temp fst file. Is deleted after the script runs
MLFST=${CURRENT_DIRECTORY}/ML.fst


# Compose M with L to create  ML.fst
python3  mkfstinput.py ${WRONG} |
    fstcompile --isymbols=${CHARSSYMS} --osymbols=${CHARSSYMS} |
    fstcompose - ${VANILLA_LEVENSHTEIN} > ${MLFST}


# Create N.fst and compose ML.fst with N.fst to create MLN.fst
python3 mkfstinput.py ${CORRECT} |
    fstcompile --isymbols=${CHARSSYMS} --osymbols=${CHARSSYMS} |
    fstcompose ${MLFST} - |
    # Run shortest path to get the edits for the minimum edit distance
    fstshortestpath |
    # Print the shortest path fst
    fstprint --isymbols=${CHARSSYMS} --osymbols=${CHARSSYMS}  --show_weight_one | grep -v "0$"| cut -d$'\t' -f3-5   
rm ${MLFST}
