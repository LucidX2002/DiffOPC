#!/usr/bin/env bash

# Use the active environment unless PYTHON is explicitly overridden.
python="${PYTHON:-python}"

start_idx=1
end_idx=10
data_dir=/home/xiaye/lucidx/DiffOPC/benchmark/edge_bench
visual_output_root=/home/xiaye/lucidx/DiffOPC/visual_outputs
dataset_name="$(basename "$data_dir")"
visual_output_dir="${visual_output_root}/${dataset_name}"

$python src/diffopc.py \
  opc=debug \
  opc.IsInsertSRAF=True \
  opc.VISUAL_DEBUG=1 \
  opc.VISUAL_OUTPUT_DIR="$visual_output_dir" \
  data=default \
  data.data_dir="$data_dir" \
  data.start_idx="$start_idx" \
  data.end_idx="$end_idx" \
  extras.print_config=false \
  logger.aim.experiment=debug_visual
