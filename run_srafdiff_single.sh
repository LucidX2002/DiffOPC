#!/usr/bin/env bash

# Use the active environment unless PYTHON is explicitly overridden.
python="${PYTHON:-python}"

data_idxs=(1 2 3 4 5 6 7 8 9 10)

# res=high

# for data_idx in "${data_idxs[@]}"; do
#     $python src/srafgen.py opc.common.resolution=$res data=sraf_single data.data_idx=$data_idx extras.print_config=false logger.aim.experiment=srafgen
#     mv tmp/grad_2048x2048 tmp/grad_2048x2048_srafgen$data_idx
# done
# # $python src/srafgen.py opc.common.resolution=$res data=sraf_single data.data_idx=$data_idx extras.print_config=true logger.aim.experiment=srafgen




# res=mid
# for data_idx in "${data_idxs[@]}"; do
#     $python src/srafgen.py opc.common.resolution=$res data=sraf_single data.data_idx=$data_idx extras.print_config=false logger.aim.experiment=srafgen
#     mv tmp/grad_1024x1024 tmp/grad_1024x1024_srafgen$data_idx
# done
# $python src/srafgen.py opc.common.resolution=$res data=sraf_single data.data_idx=$data_idx extras.print_config=true logger.aim.experiment=srafgen



data_idxs=(1)
sraf_res=low
opc_res=high
for data_idx in "${data_idxs[@]}"; do
    $python src/sraf_diffopc.py solver.sraf_resolution=$sraf_res solver.opc_resolution=$opc_res data=mscale_single data.data_idx=$data_idx extras.print_config=false logger.aim.experiment=sraf_diff
    # mv tmp/grad_1024x1024 tmp/grad_1024x1024_srafgen$data_idx
done
