# Batch Run README

这份文档专门说明当前仓库里的批量运行脚本应该怎么用，哪些脚本轻、哪些重、建议按什么顺序跑，以及哪些不要同时跑。

## 1. 开始前

先进入仓库并激活环境：

```bash
cd /home/xiaye/lucidx/DiffOPC
source activate dopc
```

建议先确认当前有没有别的长任务已经在跑：

```bash
pgrep -af "run_seg_length.sh|run_opc_iter.sh|run_srafdiff.sh|run_mrc.sh|src/diffopc.py|src/mrc/mrc.py|src/sraf_diffopc.py|src/multidiff.py|src/srafgen.py"
```

如果已经有重任务在跑，不要再同时启动另一个重任务。

## 2. 脚本分类

### A. 基础验证

- `bash run_debug.sh`
- `bash run_debug_visual.sh`

用途：
- 确认主流程能跑
- `run_debug_visual.sh` 会额外产图

### B. 轻量参数 sweep

- `bash run_seg_length.sh`
- `bash run_pvb_weight.sh`
- `bash run_sraf_tests.sh`

特点：
- 都是围绕 `src/diffopc.py` 的批量参数实验
- 适合作为第一轮 batch
- 相比大 sweep 更容易发现配置问题

### C. 中等复杂度实验

- `bash run_mscale.sh`
- `bash run_srafdiff_single.sh`
- `bash run_srafgen.sh`

特点：
- 不是最小验证，但也不是最重的全量 sweep
- 适合在基础主流程稳定后再跑

### D. 重任务

- `bash run_opc_iter.sh`
- `bash run_srafdiff.sh`
- `bash run_mrc.sh`

特点：
- 运行时间长
- 参数组合多
- 不建议和别的重任务并发

## 3. 推荐顺序

### 第一轮：确认仓库稳定

按这个顺序：

```bash
bash run_debug.sh
bash run_debug_visual.sh
bash run_seg_length.sh
bash run_pvb_weight.sh
bash run_sraf_tests.sh
```

这一轮的目标是：
- 主流程稳定
- 指标能正常输出
- 产图正常
- 轻量 sweep 不报错

### 第二轮：扩展流程

```bash
bash run_mscale.sh
bash run_srafdiff_single.sh
bash run_srafgen.sh
```

### 第三轮：重 sweep

```bash
bash run_opc_iter.sh
bash run_srafdiff.sh
```

### 第四轮：MRC

先 smoke test：

```bash
python src/mrc/mrc.py --config-name mrc_curvlarge case_count=1 min_area=5 min_wh=1 exp_name=mrc_smoke
```

确认没问题后再跑批量：

```bash
bash run_mrc.sh
```

## 4. 每个脚本大致在扫什么

### `run_seg_length.sh`

当前扫：
- `SEG_LENGTH=(40 60)`

### `run_pvb_weight.sh`

当前扫：
- `opc.WeightPVBL2=(0.2 0.5 0.7 0.9 1)`

### `run_sraf_tests.sh`

当前扫：
- `SRAF_threshold_min`
- `SRAF_FORBIDDEN`
- `SRAF_contour_area`
- `SRAF_initial_sraf_wh`

### `run_opc_iter.sh`

当前扫：
- `SEG_LENGTH=(60 80 100)`
- `Iterations=(60 70 80 90)`
- `StepSize=(1 2 4 8)`
- 同时区分带 SRAF / 不带 SRAF

这是当前仓库里更重的一类 sweep。

### `run_srafdiff.sh`

当前扫：
- `sraf_resolution`
- `opc_resolution`
- `SEG_LENGTH`
- `SRAF_FORBIDDEN`
- `SRAF_initial_sraf_wh`
- `SRAF_threshold_min`
- `max_sraf_grad_candidates`
- `WeightEPE`

### `run_mrc.sh`

当前默认扫：
- `min_area=(5 10 15 20 30 40 50)`
- `min_wh=(1 2 3 4 5 6 7 8 9 10)`

默认配置：
- `mrc_curvlarge`

## 5. 不建议同时跑的组合

不建议同时跑：
- `run_opc_iter.sh` + `run_srafdiff.sh`
- `run_opc_iter.sh` + `run_mscale.sh`
- `run_srafdiff.sh` + `run_srafgen.sh`
- `run_mrc.sh` + 任何 GPU 重任务

原因：
- `diffopc` / `sraf_diffopc` / `multidiff` 这类流程会抢 GPU
- `mrc` 主要吃 CPU，而且运行时间也长
- Aim 和日志会同时写很多输出，不利于排错

## 6. 推荐并发策略

最稳妥：
- 一次只跑一个重任务

可以接受：
- 一个轻量 sweep
- 或一个中等任务
- 或一个 MRC smoke test

如果一定要并发：
- 最多一个 GPU 主流程任务 + 一个很小的 CPU 任务
- 不要两个大 sweep 一起跑

## 7. 后台运行方式

长任务建议写日志后台跑：

```bash
mkdir -p batch_logs
nohup bash run_opc_iter.sh > batch_logs/run_opc_iter.$(date +%F-%H%M%S).log 2>&1 &
```

```bash
nohup bash run_srafdiff.sh > batch_logs/run_srafdiff.$(date +%F-%H%M%S).log 2>&1 &
```

```bash
nohup bash run_mrc.sh > batch_logs/run_mrc.$(date +%F-%H%M%S).log 2>&1 &
```

## 8. MRC 推荐跑法

不要一开始就直接全量：

```bash
python src/mrc/mrc.py --config-name mrc_curvlarge case_count=1 min_area=5 min_wh=1 exp_name=mrc_smoke
```

如果要缩小 batch 范围：

```bash
MRC_CONFIG=mrc_curvlarge MIN_AREAS="5 10 20" MIN_WHS="1 3 5" bash run_mrc.sh
```

## 9. 输出在哪里

### Hydra 日志

```text
logs/train/runs/<timestamp>/
```

### Aim

通常在项目根目录下：
- `aim/`
- `aim-mrc/`
- `aim-mrc-curvlarge/`
- 某些脚本自己的 `aim_*` 目录

### 可视化图片

如果打开 `VISUAL_DEBUG=1`，图片通常在：

```text
tmp/report_1x/...
```

### MRC 中间文件

```text
benchmark/baseline/.../rects/
```

## 10. 我给你的最实用顺序

如果你只是想稳定地批量跑一轮，直接按这个顺序：

```bash
source activate dopc
bash run_debug.sh
bash run_seg_length.sh
bash run_pvb_weight.sh
bash run_sraf_tests.sh
bash run_mscale.sh
bash run_srafdiff_single.sh
```

之后再决定是否继续跑更重的：

```bash
bash run_srafgen.sh
bash run_opc_iter.sh
bash run_srafdiff.sh
python src/mrc/mrc.py --config-name mrc_curvlarge case_count=1 min_area=5 min_wh=1 exp_name=mrc_smoke
bash run_mrc.sh
```

## 11. 现在这个仓库的现实建议

如果你现在主要目标是“先把一批实验稳定跑起来”，优先级应该是：

1. `run_seg_length.sh`
2. `run_pvb_weight.sh`
3. `run_sraf_tests.sh`
4. `run_mscale.sh`
5. `run_srafdiff_single.sh`

等这些都稳定后，再上：

1. `run_srafgen.sh`
2. `run_opc_iter.sh`
3. `run_srafdiff.sh`
4. `run_mrc.sh`
