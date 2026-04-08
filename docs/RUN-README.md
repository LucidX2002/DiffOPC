# Run README

这份文档说明当前这份仓库在本地怎么运行，重点覆盖已经在当前工作区验证过的命令，以及各入口脚本分别做什么。

## 1. 当前已验证环境

- 仓库路径：`/home/xiaye/lucidx/DiffOPC`
- 当前可用 conda 环境：`dopc`
- 当前工作区里实际验证过的 Python 版本：`3.11.12`
- 当前仓库要求的 Python 主版本：`3.11`
- 依赖安装方式：
  - 已有环境时：`conda activate dopc`
  - 新建环境时可参考 `environment.yaml` 或直接执行：

```bash
conda create -n dopc python=3.11
conda activate dopc
pip install -r requirements.txt
```

说明：
- `environment.yaml` 已经对齐到 `python=3.11`。
- `requirements.txt` 是当前公开的主依赖入口。
- 如果要把当前机器上的 `dopc` 环境重新导出成本地锁文件，执行：

```bash
bash export_dopc_env.sh
```

默认会生成：
- `requirements.lock.txt`
- `environment.lock.yaml`

## 2. 数据集位置

当前仓库没有顶层 `data/` 目录，实际数据都在 `benchmark/` 下。

最常用的数据目录：
- 默认 DiffOPC 数据：`benchmark/ICCAD2013`
- 大尺寸数据：`benchmark/ICCAD2013_large`
- 边界测试数据：`benchmark/edge_bench`
- MRC 曲线数据：`benchmark/baseline/curvmulti-large`

完整说明见 [DATASET-README.md](./DATASET-README.md)。
批量运行顺序和脚本分级见 [BATCH-RUN-README.md](./BATCH-RUN-README.md)。

## 3. 最快跑通方式

如果只是想确认项目能跑，直接执行：

```bash
cd /home/xiaye/lucidx/DiffOPC
source activate dopc
bash run_debug.sh
```

这个脚本当前会执行：

```bash
python src/diffopc.py \
  opc=debug \
  data=single \
  data.data_idx=3 \
  extras.print_config=false \
  logger.aim.experiment=debug
```

这一条命令已经在当前工作区验证通过。

如果你想默认产图，执行：

```bash
cd /home/xiaye/lucidx/DiffOPC
source activate dopc
bash run_debug_visual.sh
```

这个脚本当前默认还会开启：

```bash
opc.IsInsertSRAF=True
```

当前脚本默认等价于：

```bash
START_IDX=1
END_IDX=3
```

## 4. 主要运行入口

### 4.1 单 case DiffOPC

```bash
source activate dopc
python src/diffopc.py opc=debug data=single data.data_idx=3 extras.print_config=false logger.aim.experiment=debug
```

对应脚本：
- `bash run_debug.sh`
- `bash run_debug_visual.sh`：默认打开 `VISUAL_DEBUG=1`

默认数据：
- `benchmark/ICCAD2013`

默认产图脚本输出目录会自动包含数据集目录名，例如：
- `visual_outputs/ICCAD2013/report_sraf/...`
- `visual_outputs/ICCAD2013_large/report_sraf/...`

### 4.2 多尺度流程

```bash
source activate dopc
bash run_mscale.sh
```

对应入口：
- `src/multidiff.py`

默认数据：
- `benchmark/ICCAD2013`

### 4.3 单 case SRAF + OPC

```bash
source activate dopc
bash run_srafdiff_single.sh
```

对应入口：
- `src/sraf_diffopc.py`

默认数据：
- `benchmark/ICCAD2013`

### 4.4 SRAF 批量实验

```bash
source activate dopc
bash run_srafdiff.sh
```

说明：
- 这是批量实验脚本，会跑多组参数。
- 会比 `run_debug.sh` 慢很多。

### 4.5 SRAF 生成

```bash
source activate dopc
bash run_srafgen.sh
```

对应入口：
- `src/srafgen.py`

### 4.6 MRC

默认 MRC 脚本：

```bash
source activate dopc
bash run_mrc.sh
```

说明：
- `run_mrc.sh` 默认使用 `mrc_curvlarge` 配置。
- 它会扫很多 `min_area` 和 `min_wh` 组合，运行时间很长。

如果只想做快速 smoke test，建议直接跑 1 个 case：

```bash
source activate dopc
python src/mrc/mrc.py --config-name mrc_curvlarge case_count=1 min_area=5 min_wh=1 exp_name=mrc_smoke
```

这条命令已经在当前工作区验证通过。

## 5. 可用脚本总览

当前常用脚本：
- `run_debug.sh`：最小可用单 case 调试入口
- `run_debug_visual.sh`：单 case 调试入口，默认产图
- `run_mscale.sh`：多尺度入口
- `run_srafdiff_single.sh`：单 case 的 SRAF+OPC
- `run_srafdiff.sh`：批量 SRAF 实验
- `run_srafgen.sh`：SRAF 生成实验
- `run_mrc.sh`：MRC 参数批量扫描

这些脚本现在都已经改成优先使用当前激活环境里的 `python`，不再依赖旧机器上的绝对解释器路径。

## 6. 测试命令

仓库自带测试命令：

```bash
make test
make test-full
```

也可以直接跑 pytest：

```bash
pytest -k "not slow"
pytest
```

本地已经验证过的轻量测试：

```bash
source activate dopc
pytest -q tests/test_mrc.py
```

## 7. 运行输出在哪里

### Hydra / 主流程输出

主流程运行输出通常在：

```text
logs/train/runs/<timestamp>/
```

例如 `run_debug.sh` 的输出会落到 `logs/train/runs/...`。
`run_debug_visual.sh` 除了日志外，还会在 `tmp/report_1x/...` 下保存中间和最终图片。

### Aim 日志

Aim 实验日志会落到项目根目录下的 Aim repo 中，例如：
- `aim/`
- `aim-mrc/`
- `aim-mrc-curvlarge/`
- 某些实验脚本里自定义的 `aim_*` 目录

### MRC 中间产物

MRC 运行会在 `benchmark/baseline/.../rects/` 下写中间文件，例如：

```text
benchmark/baseline/curvmulti-large/rects/2048x2048/
benchmark/baseline/curvmulti-large/rects/2048x2048_filtered/
```

## 8. 当前已知注意事项

- 当前仓库里的真实数据在 `benchmark/`，不是 `data/`。
- `configs/paths/default.yaml` 里还有 `${PROJECT_ROOT}/data/` 这个模板路径，但当前运行主流程并不依赖它来找到 `ICCAD2013`。
- `run_mrc.sh` 默认是完整 sweep，很慢；调试时优先用 `python src/mrc/mrc.py --config-name mrc_curvlarge case_count=1 ...`。
- `MRC_CONFIG=mrc bash run_mrc.sh` 需要 `benchmark/baseline/multilevel/target`，但这个目录目前在本地缺失。
- 项目根目录里有一些未提交改动，做大范围实验前建议先确认自己的工作树状态。

## 9. 推荐使用顺序

如果你是第一次在这台机器上跑，建议按这个顺序：

```bash
cd /home/xiaye/lucidx/DiffOPC
source activate dopc
bash run_debug.sh
pytest -q tests/test_mrc.py
python src/mrc/mrc.py --config-name mrc_curvlarge case_count=1 min_area=5 min_wh=1 exp_name=mrc_smoke
```

这三步分别覆盖：
- 主流程能否正常启动
- MRC 配置和路径是否正常
- MRC 实际运行链路是否正常
