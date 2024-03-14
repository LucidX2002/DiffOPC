# Differentiable OPC

The developing framework is beased on pytorch + [hydra configuring](https://hydra.cc/) template. [FRAMEWORK README](./docs/FRAMEWORK-README.md)

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
