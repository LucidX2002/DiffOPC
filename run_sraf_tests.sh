#!/usr/bin/bash

python=/home/local/eda13/gc29434/miniconda3/envs/dopc/bin/python

# device_id=1
thres_mins=(0.1 0.2 0.3 0.4)
exp_name="thres_min"
for thres_min in "${thres_mins[@]}"; do
    $python src/diffopc.py opc.SRAF_threshold_min=$thres_min logger.aim.experiment=$exp_name
done



forbiddens=(10 20 40 60)
exp_name="sraf_forbidden"
for forbidden in "${forbiddens[@]}"; do
    $python src/diffopc.py opc.SRAF_FORBIDDEN=$forbidden logger.aim.experiment=$exp_name
done


areas=(300 400 600 900)
exp_name="sraf_area"
for area in "${areas[@]}"; do
    $python src/diffopc.py opc.SRAF_contour_area=$area logger.aim.experiment=$exp_name
done

initial_sraf_whs=(20 40 60 80)
exp_name="init_sraf_wh"
for wh in "${initial_sraf_whs[@]}"; do
    $python src/diffopc.py opc.SRAF_initial_sraf_wh=$wh logger.aim.experiment=$exp_name
done
