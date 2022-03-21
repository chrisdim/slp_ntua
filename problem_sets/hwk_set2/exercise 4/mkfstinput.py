import string
import sys
from util import EPS, format_arc

from pathlib import Path

# Read words from vocabulary
data_folder = Path("./")
file_to_open = data_folder / "corpus.txt"
words=[]
phones=[]
with open(file_to_open) as f:
    for line in f:
        words.append((line.split()[0]))
        phones.append((line.split()[1:]))


def make_input_fst(word,phone):
    """Create an fst that accepts a word letter by letter
    This can be composed with other FSTs, e.g. the spell
    checker to provide an "input" word

    """
    s, accept_state = 0, 10000
    for i in range(len(phone)):
        
        print(format_arc(s, s + 1, phone[i], phone[i], weight=0))
        s += 1

        if i == len(phone) - 1:
            print(format_arc(s, accept_state, EPS, EPS, weight=0))

    print(accept_state)


if __name__ == "__main__":
    word = sys.argv[1]
    phone=[]
    for i in range(len(words)):
        if (words[i]==word):
            phone = phones[i]
    make_input_fst(word, phone)
