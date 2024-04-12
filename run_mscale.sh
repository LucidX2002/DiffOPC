#!/usr/bin/bash

python=/home/local/eda13/gc29434/miniconda3/envs/dopc/bin/python

data_idx=1

res=high

$python src/multidiff.py opc.common.resolution=$res data=mscale_single data.data_idx=$data_idx extras.print_config=false logger.aim.experiment=mscale
