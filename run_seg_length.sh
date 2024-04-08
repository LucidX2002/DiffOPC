#!/usr/bin/bash

python=/home/local/eda13/gc29434/miniconda3/envs/dopc/bin/python

# device_id=1

seg_lengths=(40 60)
exp_name="seg_length"

for seg_length in "${seg_lengths[@]}"; do
    $python src/main.py opc.SEG_LENGTH=$seg_length opc.WeightPVBL2=0.5 logger.aim.experiment=$exp_name
done
