# Differentiable OPC

The developing framework is beased on pytorch + [hydra configuring](https://hydra.cc/) template. [FRAMEWORK README](./docs/FRAMEWORK-README.md)

Dataset layout and local dataset status are documented in [DATASET README](./docs/DATASET-README.md).
How to run the current local repo is documented in [RUN README](./docs/RUN-README.md).
Batch execution order and script guidance are documented in [BATCH RUN README](./docs/BATCH-RUN-README.md).
The local change summary relative to the upstream repo is documented in [WORKSPACE CHANGES](./docs/WORKSPACE-CHANGES.md).

## 🚀  Quickstart

```bash
# clone project
git clone https://github.com/dekura/DiffOPC.git
cd DiffOPC

# [OPTIONAL] create conda environment
conda create -n dopc python=3.10
conda activate dopc

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## Run levelset

```bash
python src/opc/levelset.py
```

## Run segments
