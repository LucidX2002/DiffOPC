# Dataset README

This document summarizes where the local datasets currently live in this workspace and which configs use them.

## Overview

- There is no top-level `data/` directory in the current checkout.
- The active local datasets live under `benchmark/`.
- The default DiffOPC runtime uses `benchmark/ICCAD2013`.

## Main Runtime Datasets

| Purpose | Local path | Configs | Status |
| --- | --- | --- | --- |
| Default DiffOPC dataset | `benchmark/ICCAD2013` | `configs/data/default.yaml`, `configs/data/single.yaml`, `configs/data/mscale.yaml`, `configs/data/mscale_single.yaml` | Present |
| Larger ICCAD benchmark set | `benchmark/ICCAD2013_large` | manual / benchmark use | Present |
| Edge-case layouts | `benchmark/edge_bench` | manual / debug use | Present |

## MRC Datasets

### `curvmulti-large`

- Mask directory: `benchmark/baseline/curvmulti-large/mask`
- Target directory: `benchmark/baseline/curvmulti-large/target_bk`
- Config: `configs/mrc/mrc_curvlarge.yaml`
- Mask filenames: `MultiLevel_mask1.png` ... `MultiLevel_mask10.png`
- Target filenames: `t11_0_mask.png` ... `t20_0_mask.png`
- Status: present in the current workspace

### `multilevel`

- Mask directory: `benchmark/baseline/multilevel/mask`
- Target directory: `benchmark/baseline/multilevel/target`
- Config: `configs/mrc/mrc.yaml`
- Expected mask filenames: `MultiLevel_mask1.png` ... `MultiLevel_mask10.png`
- Expected target filenames: `MultiLevel_target1.png` ... `MultiLevel_target10.png`
- Status: mask directory is present, but the target directory is currently missing in the local workspace

## Which Commands Use Which Dataset

- `python src/diffopc.py ...` uses `benchmark/ICCAD2013` by default.
- `bash run_debug.sh` uses `benchmark/ICCAD2013`.
- `bash run_mrc.sh` uses `mrc_curvlarge` by default, which maps to `benchmark/baseline/curvmulti-large`.
- `MRC_CONFIG=mrc bash run_mrc.sh` switches to `benchmark/baseline/multilevel`, but it still requires the missing `benchmark/baseline/multilevel/target` directory.

## Current Local Directory Layout

```text
benchmark/
  ICCAD2013/
  ICCAD2013_large/
  edge_bench/
  baseline/
    curvmulti-large/
      mask/
      rects/
      target_bk/
    multilevel/
      mask/
```

## Notes

- The `paths.data_dir` setting in `configs/paths/default.yaml` points to `${PROJECT_ROOT}/data/`, but the current project data is stored under `benchmark/` instead.
- Generated MRC intermediate files are written under `benchmark/baseline/*/rects/`.
- MRC experiment logs are written under `aim-mrc` or `aim-mrc-curvlarge` at the project root.
