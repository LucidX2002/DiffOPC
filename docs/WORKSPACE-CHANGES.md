# Workspace Changes

这份文档记录当前工作区相对于上游仓库 `dekura/DiffOPC` 的本次本地可运行性修改。

上游仓库：
- https://github.com/dekura/DiffOPC

本地基线：
- 分支：`main`
- 当前 `HEAD`：`d0462697f2cd308f8dcc8a578f188ee210708fe4`

## 结论

这份代码现在已经比原始仓库更适合在当前这台机器上直接运行。

但更准确的说法不是“所有功能都已经被完整验证”，而是：
- 主流程 `DiffOPC` 已经实际跑通
- 默认调试流程已能产生日志
- 新增的可视化调试流程已能实际产图
- `MRC` 的关键本地阻塞点已修掉，并完成了 smoke test
- 文档和数据集定位已经补齐

还没有完整验证的是：
- 所有批量实验脚本的整轮 sweep
- 所有参数组合
- 仓库里其他已有本地改动对整体行为的影响

## 原始仓库在当前机器上的主要问题

这些问题主要是“本地可运行性问题”和“环境可移植性问题”，不等于我已经证明上游算法本身错误。

### 1. 运行脚本写死了旧机器上的 Python 路径

上游脚本使用类似下面的固定解释器路径：

```bash
/home/local/eda13/gc29434/miniconda3/envs/dopc/bin/python
```

这会导致仓库在当前机器上直接运行失败，哪怕 `dopc` 环境已经存在。

受影响脚本包括：
- `run_debug.sh`
- `run_grad_test.sh`
- `run_mrc.sh`
- `run_mscale.sh`
- `run_opc_iter.sh`
- `run_pvb_weight.sh`
- `run_seg_length.sh`
- `run_sraf_tests.sh`
- `run_srafdiff.sh`
- `run_srafdiff_single.sh`
- `run_srafgen.sh`

### 2. MRC 配置写死了旧机器上的绝对数据路径

上游 `configs/mrc/*.yaml` 使用了旧机器的绝对路径，当前机器无法直接使用。

### 3. MRC 的目标文件命名假设和本地现有数据不匹配

上游 `src/mrc/mrc.py` 默认按下面的命名规则读取目标图：

```text
MultiLevel_target1.png ... MultiLevel_target10.png
```

但当前工作区里实际存在的 `curvmulti-large` 目标图是：

```text
t11_0_mask.png ... t20_0_mask.png
```

### 4. `adabox` 和当前 `scipy` 版本存在兼容性问题

在当前 `dopc` 环境中：
- `scipy == 1.15.3`

上游 `adabox` 代码依赖旧版 `scipy.stats.mode()` 的返回形状，直接运行 `MRC` 会抛：

```text
IndexError: invalid index to scalar variable
```

### 5. 仓库缺少面向当前工作区的运行文档

上游 README 只给了很简略的 Quickstart，不足以说明：
- 当前机器上的真实数据目录
- 哪些入口已经跑通
- 哪些脚本是批量实验，运行会很慢
- 哪些流程默认不会产图

## 我做了哪些修改

### A. 修复所有主要运行脚本的解释器绑定问题

修改方式：
- shebang 从 `#!/usr/bin/bash` 改为 `#!/usr/bin/env bash`
- Python 解释器从写死路径改为：

```bash
python="${PYTHON:-python}"
```

这样脚本会优先使用当前激活环境中的 `python`，也允许通过 `PYTHON=...` 手工覆盖。

修改文件：
- `run_debug.sh`
- `run_grad_test.sh`
- `run_mrc.sh`
- `run_mscale.sh`
- `run_opc_iter.sh`
- `run_pvb_weight.sh`
- `run_seg_length.sh`
- `run_sraf_tests.sh`
- `run_srafdiff.sh`
- `run_srafdiff_single.sh`
- `run_srafgen.sh`

### B. 修复 MRC 配置路径

把下面两份配置从旧机器绝对路径改成基于 `${PROJECT_ROOT}` 的可移植路径：
- `configs/mrc/mrc.yaml`
- `configs/mrc/mrc_curvlarge.yaml`

同时补充了：
- `mask_pattern`
- `target_pattern`
- `case_count`
- `mask_start_idx`
- `target_start_idx`

这样 `mrc.py` 不再只能写死读取固定文件名。

### C. 修复 MRC 的文件名映射逻辑

在 `src/mrc/mrc.py` 中增加了配置驱动的 case 文件列表生成逻辑：
- `build_case_files(...)`

现在可以同时支持：
- `MultiLevel_mask{idx}.png`
- `MultiLevel_target{idx}.png`
- `t{idx}_0_mask.png`

而不是把目标文件名硬编码死在代码里。

### D. 修复 `adabox` / `scipy` 兼容问题

在 `src/mrc/mrc.py` 中加入了本地兼容层：
- `_mode_scalar(...)`
- `compatible_get_separation_value(...)`

并把兼容函数绑定到：
- `tools.get_separation_value`
- `proc.get_separation_value`

目的是绕开 `adabox` 对旧版 `scipy.stats.mode()` 返回值形状的假设。

### E. 增强 `run_mrc.sh` 的可控性

`run_mrc.sh` 现在支持：
- `MRC_CONFIG`
- `MIN_AREAS`
- `MIN_WHS`

这样可以做更小范围的本地验证，而不是每次都直接跑完整 sweep。

### F. 新增默认产图的调试脚本

新增：
- `run_debug_visual.sh`

这个脚本默认打开：

```bash
opc.VISUAL_DEBUG=1
```

这样主流程除了写日志，还会在 `visual_outputs/<dataset>/report_sraf/...` 下输出 mask / wafer 图片。

### G. 新增本地文档

新增：
- `docs/DATASET-README.md`
- `docs/RUN-README.md`

并在 `README.md` 中加入链接，方便从主入口跳转。

### H. 新增 MRC 回归测试

新增：
- `tests/test_mrc.py`

覆盖内容：
- `mrc_curvlarge` 配置和本地目录布局是否匹配
- `mrc` 默认配置的文件名映射是否正确
- `scipy` 兼容层是否按预期工作

## 已实际验证的内容

以下命令已经在当前工作区执行并成功：

### 1. 主流程最小调试

```bash
source activate dopc
bash run_debug.sh
```

结果：
- 成功完成 `Testcase 3`
- 日志输出到 `logs/train/runs/...`

### 2. 主流程默认产图调试

```bash
source activate dopc
bash run_debug_visual.sh
```

结果：
- 成功完成 `Testcase 3`
- 生成图片目录：
  - `visual_outputs/ICCAD2013/report_sraf/M1_test3/mask/`
  - `visual_outputs/ICCAD2013/report_sraf/M1_test3/wafer/`

### 3. MRC 回归测试

```bash
source activate dopc
pytest -q tests/test_mrc.py
```

结果：
- `3 passed`

### 4. MRC smoke test

```bash
source activate dopc
python src/mrc/mrc.py --config-name mrc_curvlarge case_count=1 min_area=5 min_wh=1 exp_name=mrc_smoke
```

结果：
- 成功完成 1 个 case
- 生成过滤后图片
- 成功输出 `L2 / PVBand / EPE`

## 还没有完全证明的部分

下面这些我没有宣称“已经全部验证完”：

- `run_mrc.sh` 的完整参数 sweep 全部完成
- `run_srafdiff.sh` 这类批量实验脚本的所有组合都已跑通
- 上游所有实验入口在当前数据条件下都完整无误
- 当前仓库里其他未提交本地改动不会引入新的影响

## 关于“原始 repo 是不是有很多问题”

更准确的表述是：

- 对当前这台机器的本地运行来说，原始 repo 确实有几处明显的可运行性问题
- 这些问题主要集中在“写死机器路径”“MRC 路径和文件名假设”“依赖版本兼容”“文档不足”
- 我已经修掉了当前最主要的本地阻塞点
- 但这不等于我已经证明上游仓库的所有实验流程都完全无问题

所以当前状态可以概括为：

> 这份工作区已经达到“主流程可运行、可产图、MRC 可 smoke test、文档可用”的状态，明显比原始下载版更适合直接在当前机器上使用。

## 这次修改涉及的主要文件

- `README.md`
- `configs/mrc/mrc.yaml`
- `configs/mrc/mrc_curvlarge.yaml`
- `run_debug.sh`
- `run_grad_test.sh`
- `run_mrc.sh`
- `run_mscale.sh`
- `run_opc_iter.sh`
- `run_pvb_weight.sh`
- `run_seg_length.sh`
- `run_sraf_tests.sh`
- `run_srafdiff.sh`
- `run_srafdiff_single.sh`
- `run_srafgen.sh`
- `run_debug_visual.sh`
- `src/mrc/mrc.py`
- `tests/test_mrc.py`
- `docs/DATASET-README.md`
- `docs/RUN-README.md`

## 说明

当前工作树里还存在一些未提交改动，不全是我这次加的。

我这份文档只描述我这次为“让仓库在当前机器上更可运行”所做的修改，不对其他已有本地改动背书。
