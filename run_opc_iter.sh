#!/usr/bin/bash

python=/home/local/eda13/gc29434/miniconda3/envs/dopc/bin/python

device_id=1

iters=(50 55 60 65 70 75 80 85 90 95 100)
exp_name="iter_seg60"



for iter in "${iters[@]}"; do
    sraf_iter=$((iter + 20))
    $python src/diffopc.py solver.device_id=$device_id opc.SEG_LENGTH=60 opc.Iterations=$iter opc.SRAF_ITERATIONS=$sraf_iter logger.aim.experiment=$exp_name
done



iters=(50 55 60 65 70 75 80 85 90 95 100)
exp_name="iter_seg80"

for iter in "${iters[@]}"; do
    sraf_iter=$((iter + 20))
    $python src/diffopc.py solver.device_id=$device_id opc.SEG_LENGTH=80 opc.Iterations=$iter opc.SRAF_ITERATIONS=$sraf_iter logger.aim.experiment=$exp_name
done



iters=(50 55 60 65 70 75 80 85 90 95 100)
exp_name="iter_seg100"

for iter in "${iters[@]}"; do
    sraf_iter=$((iter + 20))
    $python src/diffopc.py solver.device_id=$device_id opc.SEG_LENGTH=100 opc.Iterations=$iter opc.SRAF_ITERATIONS=$sraf_iter logger.aim.experiment=$exp_name
done
