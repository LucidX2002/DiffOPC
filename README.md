# Differentiable OPC

This repository is based on the original [dekura/DiffOPC](https://github.com/dekura/DiffOPC) and includes Codex-assisted fixes to improve local runnability, MRC execution, scripts, and environment setup.

The project uses a Hydra-configured PyTorch workflow. The main project-specific guides are:
- [RUN README](./docs/RUN-README.md)
- [BATCH RUN README](./docs/BATCH-RUN-README.md)
- [DATASET README](./docs/DATASET-README.md)
- [WORKSPACE CHANGES](./docs/WORKSPACE-CHANGES.md)

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

## Common Entry Points

```bash
bash run_debug.sh
bash run_debug_visual.sh
bash run_mrc.sh
python src/diffopc.py opc=debug data=single data.data_idx=3
```

For more detailed run modes, dataset layout, and batch execution order, use the project docs listed above.
