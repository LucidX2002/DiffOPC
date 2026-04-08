#!/usr/bin/env bash

# Use the active environment unless PYTHON is explicitly overridden.
python="${PYTHON:-python}"

data_idx=1

res=high

$python src/multidiff.py opc.common.resolution=$res data=mscale_single data.data_idx=$data_idx extras.print_config=false logger.aim.experiment=mscale
