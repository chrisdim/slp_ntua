# Speech & Natural Language Processing - ECE NTUA 7th Semester Flow S
# PreLab2: Preparational Script
# Christos Dimopoulos - 03117037
# Dimitris Dimos - 03117165

# MUST BE EXECUTED INSIDE DIR /kaldi/egs/usc

import os
import re

# Points to current directory
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

def read_file(fname):
    with open(fname, "r") as fd:
        lines = [ln.strip().split("\n") for ln in fd.readlines()]
    return lines

def write_file(fname, data):
    with open(fname, 'w') as f:
        for item in data:
            f.write("%s\n" % item)
    
def clean_text(s):
    s = s.strip()  # strip leading / trailing spaces
    s = s.lower()  # convert to lowercase
    #s = contractions.fix(s)  # e.g. don't -> do not, you're -> you are
    s = re.sub("\s+", " ", s)  # strip multiple whitespace
    s = re.sub(r"[^a-z\s\']", " ", s)  # keep only lowercase letters and spaces
    s = s.strip()  # strip leading / trailing spaces
    return s

def tokenize(s):
    tokenized = [w for w in s.split(" ") if len(w) > 0]  # Ignore empty string

    return tokenized

def get_phones(sent):
    out = ""
    for word in sent:
      out = out + " ".join(word)+" "
    return out
        


if __name__ == "__main__":
    
    #Read transciption.txt file of sentences
    read_path = SCRIPT_DIRECTORY+"/data/transcription.txt"
    corpus = read_file(read_path)
    
    # Preprocess - Clean corpus
    clean_corpus = []
    for sentence in corpus:
        i = (clean_text(sentence[0]))
        clean_corpus.append(tokenize(i))
        
    # Read Lexicon
    read_path = SCRIPT_DIRECTORY+"/data/lexicon.txt"
    #lexicon = read_file(read_path)
    with open(read_path) as f:
        lexicon = [(line.split()) for line in f]
    for word in lexicon:
        word[0] = word[0].lower() #convert to lowercase
    
    
    # Convert to phones for each sentence
    for sentence in range(len(clean_corpus)):
        for word in range(len(clean_corpus[sentence])):
            for i in lexicon:
                if (clean_corpus[sentence][word] == i[0]):
                    clean_corpus[sentence][word] = i[1:]
                    break
    
        
    for folder in ['test', 'train', 'dev']:
        
        # Read utterids
        read_path = SCRIPT_DIRECTORY+"/data/"+folder+"/uttids"
        ids = read_file(read_path)
        
        speakers = []
        paths = []
        text = []
        for utter in ids:
            # Create uut2spk files that match ids with speakers
            spk = utter[0] + " " + utter[0][-6:-4]
            speakers.append(spk)
            # Create wav.scp files that match ids with wav directories            
            path = utter[0] + " " + SCRIPT_DIRECTORY+"/wav/"+utter[0][-6:-4]+"/"+utter[0]+".wav"
            paths.append(path)
            # Create text files that match ids with sentences as phones
            texts = utter[0] + " sil " + get_phones(clean_corpus[int(utter[0][-3:])-1]) +"sil"
            text.append(texts)
            
        # Write files
        write_path = SCRIPT_DIRECTORY+"/data/"+folder+"/utt2spk"
        write_file(write_path, speakers)
        
        write_path = SCRIPT_DIRECTORY+"/data/"+folder+"/wav.scp"
        write_file(write_path, paths)
        
        write_path = SCRIPT_DIRECTORY+"/data/"+folder+"/text"
        write_file(write_path, text)
        