#!/usr/bin/bash

python=/home/local/eda13/gc29434/miniconda3/envs/dopc/bin/python

# device_id=1

pvb_weights=(0.1 0.2 0.3 0.5)
exp_name="pvb_w"

for pvb_weight in "${pvb_weights[@]}"; do
    $python src/main.py opc.WeightPVBL2=$pvb_weight logger.aim.experiment=$exp_name
done
