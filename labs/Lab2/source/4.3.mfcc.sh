# Author: Christos Dimopoulos
. ./path.sh

# Create mfcc files for each set
. ./steps/make_mfcc.sh data/dev
. ./steps/make_mfcc.sh data/train
. ./steps/make_mfcc.sh data/test

# Calculate stats of cmvn for each set
. ./steps/compute_cmvn_stats.sh data/dev
. ./steps/compute_cmvn_stats.sh data/train
. ./steps/compute_cmvn_stats.sh data/test

