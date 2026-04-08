#!/usr/bin/env bash

# Use the active environment unless PYTHON is explicitly overridden.
python="${PYTHON:-python}"

data_idx=3

$python src/diffopc.py opc=debug data=single data.data_idx=$data_idx extras.print_config=false logger.aim.experiment=debug
