# Author: Christos Dimopoulos - 03117037
# Utilizies steps of prelab
# Must be executed inside egs/usc

mkdir data
cd data
mkdir train
mkdir dev
mkdir test
cd ..
mv lexicon.txt data/
mv transcription.txt data/
mv test_utterances.txt data/test/uttids
mv train_utterances.txt data/train/uttids
mv validation_utterances.txt data/dev/uttids

# create soft link for wavs
ln -s ./../../../slp_lab2_data/wav ./wav

# Execute python script that creates files uttids, utt2spk, wav.scp, text
# inside directories data/test data/train data/dev
# ATTENTION: 
#files trancription.txt and lexicon.txt must be inside directory /data
python3 prelab2.py

