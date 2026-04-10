#!/usr/bin/env bash

set -euo pipefail

# Use the active environment unless PYTHON is explicitly overridden.
python="${PYTHON:-python}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

data_idx="${DATA_IDX:-3}"
data_dir="${DATA_DIR:-${repo_root}/benchmark/ICCAD2013}"
visual_output_root="${VISUAL_OUTPUT_ROOT:-${repo_root}/visual_outputs}"
dataset_name="$(basename "$data_dir")"
visual_output_dir="${visual_output_root}/${dataset_name}"

$python src/diffopc.py \
  opc=debug \
  opc.IsInsertSRAF=True \
  opc.VISUAL_DEBUG=1 \
  opc.VISUAL_OUTPUT_DIR="$visual_output_dir" \
  data=single \
  data.data_dir="$data_dir" \
  data.data_idx="$data_idx" \
  extras.print_config=false \
  logger.aim.experiment=debug_visual
