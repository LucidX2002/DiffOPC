#!/usr/bin/bash

python=/home/local/eda13/gc29434/miniconda3/envs/dopc/bin/python

# device_id=1

data_idxs=(1 2 3 4 5 6 7 8 9 10)
# exp_name="sraf"

for data_idx in "${data_idxs[@]}"; do
    $python src/main.py exps=sraf data.data_idx=$data_idx
    mv tmp/grad tmp/grad_test$data_idx
done
