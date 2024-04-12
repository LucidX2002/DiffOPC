#!/usr/bin/bash

python=/home/local/eda13/gc29434/miniconda3/envs/dopc/bin/python

# device_id=1

pvb_weights=(0.2 0.5 0.7 0.9 1)
exp_name="pvb_w_sgd"

for pvb_weight in "${pvb_weights[@]}"; do
    $python src/diffopc.py opc.WeightPVBL2=$pvb_weight opc.VISUAL_DEBUG=0 opc.OPT=sgd logger.aim.experiment=$exp_name
done
