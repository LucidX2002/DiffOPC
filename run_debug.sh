#!/usr/bin/bash

python=/home/local/eda13/gc29434/miniconda3/envs/dopc/bin/python

data_idx=3

$python src/diffopc.py opc=debug data=single data.data_idx=$data_idx extras.print_config=false logger.aim.experiment=debug
