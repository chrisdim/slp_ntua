# Create FST that accepts the corpus

import string
import sys
from util import EPS, format_arc

from pathlib import Path

# Read words from corpus
data_folder = Path("./")
file_to_open = data_folder / "corpus.txt"
words=[]
phones=[]
with open(file_to_open) as f:
    for line in f:
        words.append((line.split()))

words = words[0]

for i in range(len(words)):
    print(i," ",i+1," ",words[i]," ", words[i]," 0")

print(i+1) # final state
