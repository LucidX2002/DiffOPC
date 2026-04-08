#!/usr/bin/env bash

# Use the active environment unless PYTHON is explicitly overridden.
python="${PYTHON:-python}"

# device_id=1

data_idxs=(1 2 3 4 5 6 7 8 9 10)
# exp_name="sraf"

for data_idx in "${data_idxs[@]}"; do
    $python src/diffopc.py exps=sraf data.data_idx=$data_idx
    mv tmp/grad tmp/grad_test$data_idx
done
