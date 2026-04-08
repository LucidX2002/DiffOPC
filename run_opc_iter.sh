#!/usr/bin/env bash

# Use the active environment unless PYTHON is explicitly overridden.
python="${PYTHON:-python}"

device_id=0

steps=(1 2 4 8)
iters=(60 70 80 90)

# Exp 1
exp_name="iter_seg60"
for iter in "${iters[@]}"; do
    sraf_iter=$((iter + 20))
    for step in "${steps[@]}"; do
        $python src/diffopc.py solver.device_id=$device_id opc.SEG_LENGTH=60 opc.Iterations=$iter opc.IsInsertSRAF=False opc.StepSize=$step logger.aim.experiment=$exp_name
    done
done

exp_name="iter_seg60_sraf"
for iter in "${iters[@]}"; do
    sraf_iter=$((iter + 20))
    for step in "${steps[@]}"; do
        $python src/diffopc.py solver.device_id=$device_id opc.SEG_LENGTH=60 opc.Iterations=$iter opc.SRAF_ITERATIONS=$sraf_iter opc.StepSize=$step logger.aim.experiment=$exp_name
    done
done


# Exp 2
exp_name="iter_seg80"
for iter in "${iters[@]}"; do
    sraf_iter=$((iter + 20))
    for step in "${steps[@]}"; do
        $python src/diffopc.py solver.device_id=$device_id opc.SEG_LENGTH=80 opc.Iterations=$iter opc.IsInsertSRAF=False opc.StepSize=$step logger.aim.experiment=$exp_name
    done
done

exp_name="iter_seg80_sraf"
for iter in "${iters[@]}"; do
    sraf_iter=$((iter + 20))
    for step in "${steps[@]}"; do
        $python src/diffopc.py solver.device_id=$device_id opc.SEG_LENGTH=80 opc.Iterations=$iter opc.SRAF_ITERATIONS=$sraf_iter opc.StepSize=$step logger.aim.experiment=$exp_name
    done
done


# Exp 3
exp_name="iter_seg100"

for iter in "${iters[@]}"; do
    sraf_iter=$((iter + 20))
    for step in "${steps[@]}"; do
        $python src/diffopc.py solver.device_id=$device_id opc.SEG_LENGTH=100 opc.Iterations=$iter opc.IsInsertSRAF=False opc.StepSize=$step logger.aim.experiment=$exp_name
    done
done


exp_name="iter_seg100_sraf"
for iter in "${iters[@]}"; do
    sraf_iter=$((iter + 20))
    for step in "${steps[@]}"; do
        $python src/diffopc.py solver.device_id=$device_id opc.SEG_LENGTH=100 opc.Iterations=$iter opc.SRAF_ITERATIONS=$sraf_iter opc.StepSize=$step logger.aim.experiment=$exp_name
    done
done
