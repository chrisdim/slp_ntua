# Author: Christos Dimopoulos - 03117037

. ./path.sh

# 1) Move necessary files to data/local/dict
mv extra_questions.txt lm_train.text lm_test.text lm_dev.text optional_silence.txt lexicon_phone.txt silence_phones.txt nonsilence_phones.txt data/local/dict/
cd ./data/local/dict
mv lexicon_phone.txt lexicon.txt
cd ../../../

# 2) Make temp language models
./build-lm.sh -i ./data/local/dict/lm_train.text -n 1 -o ./data/local/lm_tmp/lm_phone_ug.ilm.gz
./build-lm.sh -i ./data/local/dict/lm_train.text -n 2 -o ./data/local/lm_tmp/lm_phone_bg.ilm.gz

# 3) Make arpa files of language models

compile-lm ./data/local/lm_tmp/lm_phone_ug.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > ./data/local/nist_lm/lm_phone_ug.arpa.gz

compile-lm ./data/local/lm_tmp/lm_phone_bg.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > ./data/local/nist_lm/lm_phone_bg.arpa.gz

# 4) Create FST L of language lexicon
./prepare_lang.sh data/local/dict '<oov>' data/local/lm_tmp data/lang

# 5) Files already sorted

# 6) Create utt2spk files
utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt


# 7) Create G FST
./timit_format_data.sh
