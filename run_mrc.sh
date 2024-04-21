#!/usr/bin/bash

python=/home/local/eda13/gc29434/miniconda3/envs/dopc/bin/python


min_areas=(5 10 15 20 30 40 50)
min_whs=(1 2 3 4 5 6 7 8 9 10)

for min_area in "${min_areas[@]}"; do
    for min_wh in "${min_whs[@]}"; do
        $python src/mrc/mrc.py min_area=$min_area min_wh=$min_wh exp_name=mrc_marea_${min_area}
    done
done
