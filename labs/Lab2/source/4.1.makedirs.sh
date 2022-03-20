# Author: Christos Dimopoulos 03117037
# Must be executed in egs/usc
. ./path.sh

# Create soft links
ln -s ../wsj/s5/steps steps
ln -s ../wsj/s5/utils utils

# Create score.sh
mkdir local
cd local
ln -s ../steps/score_kaldi.sh score.sh

# Create mfcc conf
cd ..
mkdir conf
mv mfcc.conf conf/

# Make directories

cd data
mkdir lang
mkdir local
cd local
mkdir dict
mkdir lm_tmp
mkdir nist_lm


