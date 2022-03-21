# Create FST that implements Language Model

import numpy as np
from pathlib import Path

INF = 999999

# Read words from corpus
corpus = []
data_folder = Path("./")
file_to_open = data_folder / "corpus.txt"
words=[]
phones=[]
with open(file_to_open) as f:
    for line in f:
        corpus.append((line.split()))
corpus = corpus[0]

total_words = len(corpus)
bas=0
chas=0
gas=0
das=0
for w in corpus:
    if w=="ba":
        bas=bas+1
    if w=="ga":
        gas=gas+1
    if w=="cha":
        chas=chas+1
    if w=="da":
        das=das+1


# Find most common word of 2 syllables

babas, bachas, bagas, badas = 0,0,0,0
chabas, chachas, chagas, chadas = 0,0,0,0
gabas,gachas,gagas,gadas = 0,0,0,0
dabas, dachas, dagas, dadas = 0,0,0,0
for i in range(0,len(corpus)-1,2):
    if corpus[i]=="ba" and corpus[i+1] == "ba":
        babas+=1
    if corpus[i]=="ba" and corpus[i+1] == "ga":
        bagas+=1
    if corpus[i]=="ba" and corpus[i+1] == "cha":
        bachas+=1
    if corpus[i]=="ba" and corpus[i+1] == "da":
        badas+=1
        
    if corpus[i]=="ga" and corpus[i+1] == "ba":
        gabas+=1
    if corpus[i]=="ga" and corpus[i+1] == "ga":
        gagas+=1
    if corpus[i]=="ga" and corpus[i+1] == "cha":
        gachas+=1
    if corpus[i]=="ga" and corpus[i+1] == "da":
        gadas+=1
        
    if corpus[i]=="cha" and corpus[i+1] == "ba":
        chabas+=1
    if corpus[i]=="cha" and corpus[i+1] == "ga":
        chagas+=1
    if corpus[i]=="cha" and corpus[i+1] == "cha":
        chachas+=1
    if corpus[i]=="cha" and corpus[i+1] == "da":
        chadas+=1
    
    if corpus[i]=="da" and corpus[i+1] == "ba":
        dabas+=1
    if corpus[i]=="da" and corpus[i+1] == "ga":
        dagas+=1
    if corpus[i]=="da" and corpus[i+1] == "cha":
        dachas+=1
    if corpus[i]=="da" and corpus[i+1] == "da":
        dadas+=1
        
bisyllables = [babas,bagas,bachas,badas,gabas,gagas,gachas,gadas,chabas,chagas,chachas,chadas,dabas,dagas,dachas,dadas]
most_common_index = np.argmax(bisyllables)
most_common_word = bisyllables[most_common_index] #dadas 6 times

# Part (b):
for i in range(0,len(corpus)-2,3):
    if corpus[i]=="ba" and corpus[i+1] == "ba":
        babas+=1
    if corpus[i]=="ba" and corpus[i+1] == "ga":
        bagas+=1
    if corpus[i]=="ba" and corpus[i+1] == "cha":
        bachas+=1
    if corpus[i]=="ba" and corpus[i+1] == "da":
        badas+=1
        
    if corpus[i]=="ga" and corpus[i+1] == "ba":
        gabas+=1
    if corpus[i]=="ga" and corpus[i+1] == "ga":
        gagas+=1
    if corpus[i]=="ga" and corpus[i+1] == "cha":
        gachas+=1
    if corpus[i]=="ga" and corpus[i+1] == "da":
        gadas+=1
        
    if corpus[i]=="cha" and corpus[i+1] == "ba":
        chabas+=1
    if corpus[i]=="cha" and corpus[i+1] == "ga":
        chagas+=1
    if corpus[i]=="cha" and corpus[i+1] == "cha":
        chachas+=1
    if corpus[i]=="cha" and corpus[i+1] == "da":
        chadas+=1
    
    if corpus[i]=="da" and corpus[i+1] == "ba":
        dabas+=1
    if corpus[i]=="da" and corpus[i+1] == "ga":
        dagas+=1
    if corpus[i]=="da" and corpus[i+1] == "cha":
        dachas+=1
    if corpus[i]=="da" and corpus[i+1] == "da":
        dadas+=1
space = 1.5*total_words

# initial probabilities
cost_ba = (-1)*np.log(bas/space)
cost_cha = (-1)*np.log(chas/space)
cost_ga = (-1)*np.log(gas/space)
cost_da = (-1)*np.log(das/space)

# a priori
cost_baba = (-1)*np.log(babas/bas)
cost_baga = (-1)*np.log(bagas/bas)
cost_bacha = (-1)*np.log(bachas/bas)
cost_bada = (-1)*np.log(badas/bas)

cost_gaba = INF
cost_gaga = (-1)*np.log(gagas/gas)
cost_gacha = (-1)*np.log(gachas/gas)
cost_gada = (-1)*np.log(gadas/gas)

cost_chaba = (-1)*np.log(chabas/chas)
cost_chaga = (-1)*np.log(chagas/chas)
cost_chacha = INF
cost_chada = (-1)*np.log(chadas/chas)

cost_daba = (-1)*np.log(dabas/das)
cost_daga = (-1)*np.log(dagas/das)
cost_dacha = (-1)*np.log(dachas/das)
cost_dada = (-1)*np.log(dadas/das)

print("0 0 ba ba ",cost_ba)
print("0 0 cha cha ",cost_cha)
print("0 0 da da ",cost_da)
print("0 0 ga ga ",cost_ga)

print("0 1 ba <eps> 0")
print("1 0 ba baba ", cost_baba)
print("1 0 cha bacha ", cost_bacha)
print("1 0 da bada ", cost_bada)
print("1 0 ga baga ", cost_baga)

print("0 2 cha <eps> 0")
print("2 0 ba chaba ", cost_chaba)
print("2 0 cha chacha ",cost_chacha)
print("2 0 da chada ", cost_chada)
print("2 0 ga chaga ", cost_chaga)

print("0 3 da <eps> 0")
print("3 0 ba daba ", cost_daba)
print("3 0 cha dacha ",cost_dacha)
print("3 0 da dada ", cost_dada)
print("3 0 ga daga ", cost_daga)

print("0 4 ga <eps> 0")
print("4 0 ba gaba ", cost_gaba)
print("4 0 cha gacha ",cost_gacha)
print("4 0 da gada ", cost_gada)
print("4 0 ga gaga ", cost_gaga)

print("0") # final state