#!/usr/bin/env bash

set -euo pipefail

env_name="${DOPC_ENV_NAME:-dopc}"
requirements_out="${DOPC_REQUIREMENTS_OUTPUT:-requirements-dopc.txt}"
conda_out="${DOPC_CONDA_OUTPUT:-environment-dopc.yaml}"
conda_bin="${CONDA_BIN:-conda}"

pip_cmd=("$conda_bin" run -n "$env_name" python -m pip freeze --exclude-editable)
conda_cmd=("$conda_bin" env export -n "$env_name")

if [[ "${DOPC_EXPORT_FROM_HISTORY:-0}" == "1" ]]; then
    conda_cmd+=("--from-history")
fi

if [[ "${DOPC_EXPORT_DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${pip_cmd[@]}"
    printf '> %q\n' "$requirements_out"
    printf '%q ' "${conda_cmd[@]}"
    printf '> %q\n' "$conda_out"
    exit 0
fi

"${pip_cmd[@]}" > "$requirements_out"
"${conda_cmd[@]}" > "$conda_out"

echo "Exported ${env_name} pip requirements to ${requirements_out}"
echo "Exported ${env_name} conda environment to ${conda_out}"
