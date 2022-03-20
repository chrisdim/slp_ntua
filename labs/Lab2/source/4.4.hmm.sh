# Author: Christos Dimopoulos - 03117037
. ./path.sh
echo This may take a while.
# 1) Train GMM-HMM acoustic model over train data
cp data/lexicon.txt data/lang/lexicon.txt

./steps/train_mono.sh data/train data/lang exp/mono 

# 2) Create HCLG Graph
utils/mkgraph.sh data/lang_phones_ug exp/mono exp/mono_graph_ug
utils/mkgraph.sh data/lang_phones_bg exp/mono exp/mono_graph_bg

# 3) Viterbi Algorithm: Decode test and dev
# 4) Evaluate model with PER Parameter

./steps/decode.sh exp/mono_graph_ug data/dev exp/mono/decode_dev_ug
./steps/decode.sh exp/mono_graph_bg data/dev exp/mono/decode_dev_bg
./steps/decode.sh exp/mono_graph_ug data/test exp/mono/decode_test_ug
./steps/decode.sh exp/mono_graph_bg data/test exp/mono/decode_test_bg

# 5) Monophone Allignment and Triphone Training
./steps/align_si.sh data/train data/lang exp/mono exp/mono_ali
./steps/train_deltas.sh 2000 10000 data/train data/lang exp/mono_ali exp/tri1

# Compute Graph and Decode for triphone
./utils/mkgraph.sh data/lang_phones_ug exp/tri1 exp/tri1_graph_ug
./utils/mkgraph.sh data/lang_phones_bg exp/tri1 exp/tri1_graph_bg

./steps/decode.sh exp/tri1_graph_ug data/dev exp/tri1/decode_dev_ug
./steps/decode.sh exp/tri1_graph_bg data/dev exp/tri1/decode_dev_bg
./steps/decode.sh exp/tri1_graph_ug data/test exp/tri1/decode_test_ug
./steps/decode.sh exp/tri1_graph_bg data/test exp/tri1/decode_test_bg


