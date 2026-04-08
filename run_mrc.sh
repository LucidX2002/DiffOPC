#!/usr/bin/env bash

# Use the active environment unless PYTHON is explicitly overridden.
python="${PYTHON:-python}"

config_name="${MRC_CONFIG:-mrc_curvlarge}"
min_areas=(${MIN_AREAS:-"5 10 15 20 30 40 50"})
min_whs=(${MIN_WHS:-"1 2 3 4 5 6 7 8 9 10"})

for min_area in "${min_areas[@]}"; do
    for min_wh in "${min_whs[@]}"; do
        $python src/mrc/mrc.py --config-name "$config_name" min_area=$min_area min_wh=$min_wh exp_name=mrc_marea_${min_area}_mwh_${min_wh}
    done
done
