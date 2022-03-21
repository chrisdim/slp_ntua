# Source: https://buipalsulich.com/post/levenshtein-edit-distance-with-fsts/
import string
import sys

alphabet = list(string.ascii_lowercase)
weight = {
    "delete": 1.3,
    "insert": 1.3,
    "sub": 1.0
}

#Keep only alphabet consisting of 'ab' to draw FST
alphabet = ['eh','er', 'iy', 'm', 'n', 'p', 's', 't', 'uh', 'uw', 'z']

L = []
# No edit
for l in alphabet:
    L.append(("0 0 %s %s %.1f" % (l, l, 0)))
 
# Deletes: input character, output epsilon
for l in alphabet:
    L.append(("0 0 %s <eps> %.1f" % (l, weight["delete"])))

# Insertions: input epsilon, output character
for l in alphabet:
    L.append("0 0 <eps> %s %.1f" % (l, weight["insert"]))

# Substitutions: input one character, output another
for l in alphabet:
    for r in alphabet:
        if l is not r:
            if (l=='uh' and r=='ur') or (r=='uh' and l=='ur') or (l=='eh' and r=='er') or (r=='eh' and l=='er') or (l=='p' and r=='t') or (r=='p' and l=='t') or (l=='m' and r=='n') or (r=='m' and l=='n') or (l=='s' and r=='z') or (r=='s' and l=='z'):
                L.append("0 0 %s %s %.1f" % (l, r, 0.5*weight["sub"]))
                continue
            L.append("0 0 %s %s %.1f" % (l, r, weight["sub"]))

# Final state
L.append(0)

from pathlib import Path
data_folder = Path("./")
file_to_open = data_folder / "L.txt"

with open(file_to_open, 'w') as f:
    for i in range(len(L)):
        string = str(L[i])+ "\n"
        f.write(string)


