#!/usr/bin/env bash

# Use the active environment unless PYTHON is explicitly overridden.
python="${PYTHON:-python}"

# device_id=1

seg_lengths=(40 60)
exp_name="seg_length"

for seg_length in "${seg_lengths[@]}"; do
    $python src/diffopc.py opc.SEG_LENGTH=$seg_length opc.WeightPVBL2=0.5 logger.aim.experiment=$exp_name
done
