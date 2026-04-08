# Differentiable OPC

This repository is based on the original [dekura/DiffOPC](https://github.com/dekura/DiffOPC) and includes Codex-assisted fixes to improve local runnability, MRC execution, scripts, and environment setup.

The developing framework is beased on pytorch + [hydra configuring](https://hydra.cc/) template. [FRAMEWORK README](./docs/FRAMEWORK-README.md)

Dataset layout and local dataset status are documented in [DATASET README](./docs/DATASET-README.md).
How to run the current local repo is documented in [RUN README](./docs/RUN-README.md).
Batch execution order and script guidance are documented in [BATCH RUN README](./docs/BATCH-RUN-README.md).
The local change summary relative to the upstream repo is documented in [WORKSPACE CHANGES](./docs/WORKSPACE-CHANGES.md).

## 🚀  Quickstart

```bash
# clone project
git clone https://github.com/LucidX2002/DiffOPC.git
cd DiffOPC

# [OPTIONAL] create conda environment
conda create -n dopc python=3.11
conda activate dopc

# install requirements
pip install -r requirements.txt
```

## Run levelset

```bash
python src/opc/levelset.py
```

## Run segments
